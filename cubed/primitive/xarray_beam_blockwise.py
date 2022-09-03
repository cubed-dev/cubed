from collections import defaultdict

import apache_beam as beam
import toolz
import xarray as xr
import xarray_beam as xbeam
import zarr
from dask.array.core import normalize_chunks
from dask.blockwise import broadcast_dimensions, lol_tuples
from dask.core import flatten
from dask.utils import apply
from xarray_beam._src.core import (
    _chunks_to_offsets,
    compute_offset_index,
    normalize_expanded_chunks,
)

from cubed.runtime.utils import gensym
from cubed.utils import chunk_memory, to_chunksize


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
    pcollection: PCollection representing the rechunked array
    target:  ArrayProxy for the Zarr array output
    required_mem: minimum memory required per-task, in bytes
    """

    # Use dask's make_blockwise_graph
    arrays = args[::2]
    array_names = [array.name for array in arrays]

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

    chunksize = to_chunksize(chunks)
    if isinstance(target_store, zarr.Array):
        target_array = target_store
    else:
        target_array = zarr.empty(
            shape, store=target_store, chunks=chunksize, dtype=dtype
        )

    # calculate memory requirement
    required_mem = 0
    # inputs
    for array in arrays:  # inputs
        # memory for a compressed and an uncompressed input array chunk
        # - we assume compression has no effect (so it's an overestimate)
        # - ideally we'd be able to look at nbytes_stored,
        #   but this is not possible in general since the array has not been written yet
        required_mem += chunk_memory(array.dtype, to_chunksize(array.chunks)) * 2
    # output
    # memory for a compressed and an uncompressed output array chunk
    # - this assumes the blockwise function creates a new array)
    # - numcodecs uses a working output buffer that's the size of the array being compressed
    required_mem += chunk_memory(dtype, chunksize) * 2

    if required_mem > max_mem:
        raise ValueError(
            f"Blockwise memory ({required_mem}) exceeds max_mem ({max_mem})"
        )

    block_fn = make_inverse_blockwise_function_from_graph(
        "out", out_ind, *argindsstr, numblocks=numblocks, new_axes=new_axes
    )
    in_shape_map = {a.name: a.shape for a in arrays}
    in_chunks_map = {a.name: a.chunks for a in arrays}
    xbeam_block_func = xbeam_key_wrapper(
        block_fn, in_shape_map, in_chunks_map, shape, chunks
    )

    pipeline = arrays[0].plan.beam_pipeline
    input_pcollections = {a.name: a.plan.pcollections[a.name] for a in arrays}

    class MapKeys(beam.DoFn):
        def __init__(self, xbeam_block_func, array_name):
            self.xbeam_block_func = xbeam_block_func
            self.array_name = array_name

        def process(self, obj):
            k, v = obj
            for out in self.xbeam_block_func(self.array_name, k):
                yield (out, v)

    pcollections = {}

    for array in arrays:
        new_pcollection = input_pcollections[array.name] | gensym(
            "map_keys"
        ) >> beam.ParDo(MapKeys(xbeam_block_func, array.name))
        pcollections[array.name] = new_pcollection

    array_names = [array.name for array in arrays]

    dims = tuple(f"dim_{i}" for i in range(len(shape)))

    def apply_func(indexed_dict):
        idx, dict = indexed_dict
        args = [dict[name][0] for name in array_names]
        # convert from dataset to a numpy array
        args = [arg["a"].values for arg in args]
        # call function
        res = func(*args, **kwargs)
        # convert back to a dataset
        ds = xr.Dataset({"a": (dims, res)})
        return idx, ds

    new_pcollection = (
        pcollections
        | gensym("cogroup") >> beam.CoGroupByKey()
        | gensym(func.__name__ if hasattr(func, "__name__") else "func")
        >> beam.Map(apply_func)
    )

    num_tasks = 1  # TODO

    return pipeline, new_pcollection, target_array, required_mem, num_tasks


def make_inverse_blockwise_function_from_graph(
    output, out_indices, *arrind_pairs, numblocks=None, new_axes=None
):
    """Make a function that is the inverse of make_blockwise_graph."""
    from dask.blockwise import make_blockwise_graph

    graph = make_blockwise_graph(
        lambda x: 0,
        output,
        out_indices,
        *arrind_pairs,
        numblocks=numblocks,
        new_axes=new_axes,
    )
    inverses = invert_blockwise_graph(graph)

    def blockwise_fn(arg, keytup):
        return inverses[arg][keytup]

    return blockwise_fn


def invert_blockwise_graph(graph):
    """Invert the graph so it maps input indexes to output indexes"""
    inverses = {}
    for key in graph.keys():  # key is like ('z', 0, 1)
        out_idx = key[1:]  # out_idx is like (0, 1)

        # inputs is like (('x', 0, 1), ('y', 0, 1))
        inputs = graph[key][2:] if graph[key][0] == apply else graph[key][1:]

        # remove any nested structure due to contraction
        # we only care about how an input index maps to an output index
        if not isinstance(inputs[0], list):
            inputs = [[v] for v in inputs]
        inputs = list(flatten(inputs))

        for input in inputs:  # input is like ('x', 0, 1)
            in_name = input[0]  # in_name is like 'x'
            in_idx = input[1:]  # in_name is like (0, 1)
            if in_name not in inverses:
                inverses[in_name] = defaultdict(list)
            inverses[in_name][in_idx].append(out_idx)

    return inverses


def make_inverse_blockwise_function_dynamic(
    output, out_indices, *arrind_pairs, numblocks=None, new_axes=None
):
    """Make a function that is a dynamic equivalent to the inverse of make_blockwise_graph.

    This is useful since it has a smaller serialization size.

    Note: currently not used since some edge cases don't work!
    """
    if numblocks is None:
        raise ValueError("Missing required numblocks argument.")
    new_axes = new_axes or {}

    argpairs = list(toolz.partition(2, arrind_pairs))

    assert set(numblocks) == {name for name, ind in argpairs if ind is not None}

    all_indices = {x for _, ind in argpairs if ind for x in ind}

    dims = broadcast_dimensions(argpairs, numblocks)
    for k, v in new_axes.items():
        dims[k] = len(v) if isinstance(v, tuple) else 1

    argdict = {argpair[0]: argpair[1] for argpair in argpairs}

    def blockwise_fn(arg, keytup):
        # Given a key for an arg (e.g. arg="x", keytup=(1, 0))
        # produce all the output keys
        keydict = dict(zip(out_indices, keytup))

        # find which indices were contracted
        dummy_indices = set(all_indices) - keydict.keys()
        # ... which ones were broadcast
        for i, x in enumerate(argdict[arg]):
            if dims[x] != numblocks[arg][i]:
                dummy_indices.add(x)

        dummies = dict((i, list(range(dims[i]))) for i in dummy_indices)

        ind = argdict[arg]

        lol = lol_tuples((output,), ind, keydict, dummies)
        if isinstance(lol, tuple):
            return [lol[1:]]
        else:
            return [lst[1:] for lst in lol]

    return blockwise_fn


def xbeam_key_wrapper(blockwise_fn, in_shape_map, in_chunks_map, out_shape, out_chunks):
    """Wrapper function that converts Xarray-Beam keys in inputs to chunks indexes,
    applies the blockwise function, then converts the output chunk indexes to keys.

    Note that the key-index mappings depend on the array shape and chunks, so must
    be applied separately for each input and output.
    """
    _, index_to_offset = xarray_beam_key_mappings(out_shape, out_chunks)

    offset_to_index_map = {}
    for arg, shape in in_shape_map.items():
        chunks = in_chunks_map[arg]
        offset_to_index, _ = xarray_beam_key_mappings(shape, chunks)
        offset_to_index_map[arg] = offset_to_index

    def wrapper(arg, key):
        keytup = xbeam_key_to_index(key, offset_to_index_map[arg])
        res = blockwise_fn(arg, keytup)
        return [index_to_xbeam_key(idx, index_to_offset) for idx in res]

    return wrapper


# Functions to convert between Xarray-Beam Key objects and chunk indexes


def get_dims(ndim):
    return tuple(f"dim_{i}" for i in range(ndim))


def xarray_beam_key_mappings(shape, chunks):
    # Use xarray-beam utilities to find the mapping from chunk index to offset for an array with given shape and chunks
    dims = get_dims(len(shape))
    dim_sizes = dict(zip(dims, shape))
    xchunks = dict(zip(dims, to_chunksize(normalize_chunks(chunks, shape))))
    expanded_chunks = normalize_expanded_chunks(xchunks, dim_sizes)
    offsets = _chunks_to_offsets(expanded_chunks)
    offset_to_index = compute_offset_index(offsets)
    index_to_offset = {k: _invert_dict(v) for k, v in offset_to_index.items()}
    return offset_to_index, index_to_offset


def _invert_dict(d):
    return {v: k for k, v in d.items()}


def get_offset_to_index(shape, chunks):
    return xarray_beam_key_mappings(shape, chunks)[0]


def get_index_to_offset(shape, chunks):
    return xarray_beam_key_mappings(shape, chunks)[1]


def index_to_xbeam_key(idx, index_to_offset):
    dims = get_dims(len(idx))
    offsets = {dims[i]: index_to_offset[dims[i]][ix] for i, ix in enumerate(idx)}
    return xbeam.Key(offsets)


def xbeam_key_to_index(key, offset_to_index):
    dims = get_dims(len(key.offsets))
    return tuple(offset_to_index[dim][key.offsets[dim]] for dim in dims)
