from functools import partial
from typing import Callable, Dict, NamedTuple

import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.blockwise import make_blockwise_graph
from dask.core import flatten
from rechunker.types import ArrayProxy, Pipeline, Stage
from toolz import map

from cubed.utils import chunk_memory, get_item, to_chunksize

sym_counter = 0


def gensym(name):
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


class BlockwiseSpec(NamedTuple):
    function: Callable
    reads_map: Dict[str, ArrayProxy]
    write: ArrayProxy


def apply_blockwise(graph_item, *, config=BlockwiseSpec):
    out_chunk_key, in_name_chunk_keys = graph_item
    args = []
    for name, in_chunk_key in in_name_chunk_keys:
        arg = config.reads_map[name].array[in_chunk_key]
        args.append(arg)
    config.write.array[out_chunk_key] = config.function(*args)


def apply_blockwise_structured(graph_item, *, config=BlockwiseSpec):
    out_chunk_key, in_name_chunk_keys = graph_item
    args = []
    for name, in_chunk_key in in_name_chunk_keys:
        arg = np.asarray(config.reads_map[name].array[in_chunk_key])
        args.append(arg)
    result = config.function(*args)
    for k, v in result.items():
        config.write.array.set_basic_selection(out_chunk_key, v, fields=k)


def blockwise(
    func,
    out_ind,
    *args,
    max_mem,
    target_store,
    shape,
    dtype,
    chunks,
    new_axes=None,
    **kwargs,
):
    """Apply a function across blocks from multiple source Zarr arrays.

    Parameters
    ----------
    func : callable
        Function to apply to individual tuples of blocks
    out_ind : iterable
        Block pattern of the output, something like 'ijk' or (1, 2, 3)
    *args : sequence of Array, index pairs
        Sequence like (x, 'ij', y, 'jk', z, 'i')
    max_mem : int
        Maximum memory allowed for a single task of this operation, measured in bytes
    target_store : string or zarr.Array
        Path to output Zarr store, or Zarr array
    shape : tuple
        The shape of the output array.
    dtype : np.dtype
        The ``dtype`` of the output array.
    chunks : tuple
        The chunks of the output array.
    new_axes : dict
        New indexes and their dimension lengths
    **kwargs : dict
        Extra keyword arguments to pass to function

    Returns
    -------
    pipeline:  Pipeline to run the operation
    target:  ArrayProxy for the Zarr array output
    required_mem: minimum memory required per-task, in bytes
    """

    # Use dask's make_blockwise_graph
    arrays = args[::2]
    array_names = [f"in_{i}" for i in range(len(arrays))]
    array_map = {name: array for name, array in zip(array_names, arrays)}

    inds = args[1::2]

    numblocks = {}
    for name, array in zip(array_names, arrays):
        input_chunks = normalize_chunks(
            array.chunks, shape=array.shape, dtype=array.dtype
        )
        numblocks[name] = tuple(map(len, input_chunks))

    argindsstr = []
    for name, ind in zip(array_names, inds):
        argindsstr.extend((name, ind))

    # TODO: check output shape and chunks are consistent with inputs
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)

    graph = make_blockwise_graph(
        func, "out", out_ind, *argindsstr, numblocks=numblocks, new_axes=new_axes
    )

    # convert chunk indexes to chunk keys (slices)
    graph_mappable = []
    for k, v in graph.items():
        out_chunk_key = get_item(chunks, k[1:])
        name_chunk_keys = []
        name_chunk_inds = v[1:]  # remove func
        # flatten (nested) lists indicating contraction
        if isinstance(name_chunk_inds[0], list):
            name_chunk_inds = list(flatten(name_chunk_inds))
        for name_chunk_ind in name_chunk_inds:
            name = name_chunk_ind[0]
            chunk_ind = name_chunk_ind[1:]
            arr = array_map[name]
            chks = normalize_chunks(
                arr.chunks, shape=arr.shape, dtype=arr.dtype
            )  # have to normalize zarr chunks
            chunk_key = get_item(chks, chunk_ind)
            name_chunk_keys.append((name, chunk_key))
        # following has to be a list, not a tuple, otherwise lithops breaks
        graph_mappable.append([out_chunk_key, name_chunk_keys])

    # now use the graph_mappable in a pipeline

    chunksize = to_chunksize(chunks)
    if isinstance(target_store, zarr.Array):
        target_array = target_store
    else:
        target_array = zarr.empty(
            shape, store=target_store, chunks=chunksize, dtype=dtype
        )

    func_with_kwargs = partial(func, **kwargs)
    read_proxies = {
        name: ArrayProxy(array, array.chunks) for name, array in array_map.items()
    }
    write_proxy = ArrayProxy(target_array, chunksize)
    spec = BlockwiseSpec(func_with_kwargs, read_proxies, write_proxy)

    if np.dtype(dtype).fields is None:
        apply_blockwise_func = apply_blockwise
    else:
        apply_blockwise_func = apply_blockwise_structured
    stages = [
        Stage(
            apply_blockwise_func,
            gensym("apply_blockwise"),
            mappable=graph_mappable,
        )
    ]

    # calculate memory requirement
    required_mem = 0
    # inputs
    for array in arrays:  # inputs
        # memory for a compressed and an uncompressed input array chunk
        # - we assume compression has no effect (so it's an overestimate)
        # - ideally we'd be able to look at nbytes_stored,
        #   but this is not possible in general since the array has not been written yet
        required_mem += chunk_memory(array.dtype, array.chunks) * 2
    # output
    # memory for a compressed and an uncompressed output array chunk
    # - this assumes the blockwise function creates a new array)
    # - numcodecs uses a working output buffer that's the size of the array being compressed
    required_mem += chunk_memory(dtype, chunksize) * 2

    if required_mem > max_mem:
        raise ValueError(
            f"Blockwise memory ({required_mem}) exceeds max_mem ({max_mem})"
        )

    num_tasks = len(graph)

    return Pipeline(stages, config=spec), target_array, required_mem, num_tasks


# Code for fusing pipelines


def is_fuse_candidate(node_dict):
    """
    Return True if the array for a node is a candidate for map fusion.
    """
    pipeline = node_dict.get("pipeline", None)
    if pipeline is None:
        return False

    stages = pipeline.stages
    return len(stages) == 1 and stages[0].function == apply_blockwise


def can_fuse_pipelines(n1_dict, n2_dict):
    if is_fuse_candidate(n1_dict) and is_fuse_candidate(n2_dict):
        return n1_dict["num_tasks"] == n2_dict["num_tasks"]
    return False


def fuse(
    pipeline1,
    target_array1,
    required_mem1,
    num_tasks1,
    pipeline2,
    target_array2,
    required_mem2,
    num_tasks2,
):
    """
    Fuse two blockwise pipelines into a single pipeline, avoiding writing to (or reading from) the target of the first pipeline.
    """
    assert num_tasks1 == num_tasks2

    # combine mappables by using input keys for first pipeline, and output keys for second
    out_keys1, in_keys1 = list(zip(*pipeline1.stages[0].mappable))
    out_keys2, in_keys2 = list(zip(*pipeline2.stages[0].mappable))
    # following has to be a list of lists, not of tuples, otherwise lithops breaks
    mappable = [list(a) for a in zip(out_keys2, in_keys1)]

    stages = [
        Stage(
            apply_blockwise,
            gensym("fused_apply_blockwise"),
            mappable=mappable,
        )
    ]

    def fused_func(*args):
        return pipeline2.config.function(pipeline1.config.function(*args))

    read_proxies = pipeline1.config.reads_map
    write_proxy = pipeline2.config.write
    spec = BlockwiseSpec(fused_func, read_proxies, write_proxy)
    pipeline = Pipeline(stages, config=spec)

    target_array = target_array2
    required_mem = max(required_mem1, required_mem2)
    num_tasks = num_tasks2

    return pipeline, target_array, required_mem, num_tasks
