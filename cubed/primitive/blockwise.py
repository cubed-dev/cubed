import itertools
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict

import toolz
import zarr
from dask.array.core import normalize_chunks
from dask.blockwise import _get_coord_mapping, _make_dims, lol_product
from dask.core import flatten
from rechunker.types import ArrayProxy, Stage
from toolz import map

from cubed.utils import chunk_memory, get_item, to_chunksize

from .types import CubedPipeline

sym_counter = 0


def gensym(name):
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


@dataclass(frozen=True)
class BlockwiseSpec:
    """Specification for how to run blockwise on an array.

    This is similar to ``CopySpec`` in rechunker.

    Attributes
    ----------
    block_function : Callable
        A function that maps input chunk indexes to an output chunk index.
    function : Callable
        A function that maps input chunks to an output chunk.
    reads_map : Dict[str, ArrayProxy]
        Read proxy dictionary keyed by array name.
    write : ArrayProxy
        Write proxy with an ``array`` attribute that supports ``__setitem__``.
    """

    block_function: Callable
    function: Callable
    reads_map: Dict[str, ArrayProxy]
    write: ArrayProxy


def apply_blockwise(out_key, *, config=BlockwiseSpec):
    """Stage function for blockwise."""
    # lithops needs params to be lists not tuples, so convert back
    out_key = tuple(out_key)
    out_chunk_key = key_to_slices(out_key, config.write.array, config.write.chunks)
    args = []
    name_chunk_inds = config.block_function(("out",) + out_key)
    for name_chunk_ind in name_chunk_inds:
        name = name_chunk_ind[0]
        chunk_ind = name_chunk_ind[1:]
        arr = config.reads_map[name].array
        chunk_key = key_to_slices(chunk_ind, arr)
        arg = arr[chunk_key]
        args.append(arg)

    result = config.function(*args)
    if isinstance(result, dict):  # structured array with named fields
        for k, v in result.items():
            config.write.array.set_basic_selection(out_chunk_key, v, fields=k)
    else:
        config.write.array[out_chunk_key] = result


def key_to_slices(key, arr, chunks=None):
    """Convert a chunk index key to a tuple of slices"""
    chunks = normalize_chunks(chunks or arr.chunks, shape=arr.shape, dtype=arr.dtype)
    return get_item(chunks, key)


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
    in_names=None,
    out_name=None,
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
    array_names = in_names or [f"in_{i}" for i in range(len(arrays))]
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

    # block func

    block_function = make_blockwise_function_flattened(
        func,
        out_name or "out",
        out_ind,
        *argindsstr,
        numblocks=numblocks,
        new_axes=new_axes,
    )

    output_blocks = get_output_blocks(
        func,
        out_name or "out",
        out_ind,
        *argindsstr,
        numblocks=numblocks,
        new_axes=new_axes,
    )

    # end block func

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
    spec = BlockwiseSpec(block_function, func_with_kwargs, read_proxies, write_proxy)

    stages = [
        Stage(
            apply_blockwise,
            gensym("apply_blockwise"),
            mappable=output_blocks,
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

    num_tasks = len(output_blocks)

    return CubedPipeline(stages, spec, target_array, required_mem, num_tasks)


# Code for fusing pipelines


def is_fuse_candidate(pipeline):
    """
    Return True if a pipeline is a candidate for blockwise fusion.
    """
    stages = pipeline.stages
    return len(stages) == 1 and stages[0].function == apply_blockwise


def can_fuse_pipelines(pipeline1, pipeline2):
    if is_fuse_candidate(pipeline1) and is_fuse_candidate(pipeline2):
        return pipeline1.num_tasks == pipeline2.num_tasks
    return False


def fuse(pipeline1, pipeline2):
    """
    Fuse two blockwise pipelines into a single pipeline, avoiding writing to (or reading from) the target of the first pipeline.
    """

    assert pipeline1.num_tasks == pipeline2.num_tasks

    mappable = pipeline2.stages[0].mappable

    stages = [
        Stage(
            apply_blockwise,
            gensym("fused_apply_blockwise"),
            mappable=mappable,
        )
    ]

    def fused_blockwise_func(out_key):
        return pipeline1.config.block_function(
            *pipeline2.config.block_function(out_key)
        )

    def fused_func(*args):
        return pipeline2.config.function(pipeline1.config.function(*args))

    read_proxies = pipeline1.config.reads_map
    write_proxy = pipeline2.config.write
    spec = BlockwiseSpec(fused_blockwise_func, fused_func, read_proxies, write_proxy)

    target_array = pipeline2.target_array
    required_mem = max(pipeline1.required_mem, pipeline2.required_mem)
    num_tasks = pipeline2.num_tasks

    return CubedPipeline(stages, spec, target_array, required_mem, num_tasks)


# blockwise functions


def make_blockwise_function(
    func, output, out_indices, *arrind_pairs, numblocks=None, new_axes=None
):
    """Make a function that is the equivalent of make_blockwise_graph."""

    if numblocks is None:
        raise ValueError("Missing required numblocks argument.")
    new_axes = new_axes or {}
    argpairs = list(toolz.partition(2, arrind_pairs))

    # Dictionary mapping {i: 3, j: 4, ...} for i, j, ... the dimensions
    dims = _make_dims(argpairs, numblocks, new_axes)

    # Generate the abstract "plan" before constructing
    # the actual graph
    (coord_maps, concat_axes, dummies) = _get_coord_mapping(
        dims,
        output,
        out_indices,
        numblocks,
        argpairs,
        False,
    )

    def blockwise_fn(out_key):
        out_coords = out_key[1:]

        # from Dask make_blockwise_graph
        deps = set()
        coords = out_coords + dummies
        args = []
        for cmap, axes, (arg, ind) in zip(coord_maps, concat_axes, argpairs):
            if ind is None:
                args.append(arg)
            else:
                arg_coords = tuple(coords[c] for c in cmap)
                if axes:
                    tups = lol_product((arg,), arg_coords)
                    deps.update(flatten(tups))
                else:
                    tups = (arg,) + arg_coords
                    deps.add(tups)
                args.append(tups)

        args.insert(0, func)
        val = tuple(args)
        # end from make_blockwise_graph

        return val

    return blockwise_fn


def make_blockwise_function_flattened(
    func, output, out_indices, *arrind_pairs, numblocks=None, new_axes=None
):
    # TODO: make this a part of make_blockwise_function?
    blockwise_fn = make_blockwise_function(
        func, output, out_indices, *arrind_pairs, numblocks=numblocks, new_axes=new_axes
    )

    def blockwise_fn_flattened(out_key):
        name_chunk_inds = blockwise_fn(out_key)[1:]  # drop function in position 0
        # flatten (nested) lists indicating contraction
        if isinstance(name_chunk_inds[0], list):
            name_chunk_inds = list(flatten(name_chunk_inds))
        return name_chunk_inds

    return blockwise_fn_flattened


def get_output_blocks(
    func, output, out_indices, *arrind_pairs, numblocks=None, new_axes=None
):

    if numblocks is None:
        raise ValueError("Missing required numblocks argument.")
    new_axes = new_axes or {}
    argpairs = list(toolz.partition(2, arrind_pairs))

    # Dictionary mapping {i: 3, j: 4, ...} for i, j, ... the dimensions
    dims = _make_dims(argpairs, numblocks, new_axes)

    # return a list of lists, not of tuples, otherwise lithops breaks
    output_blocks_lists = [
        list(tup) for tup in itertools.product(*[range(dims[i]) for i in out_indices])
    ]
    return output_blocks_lists
