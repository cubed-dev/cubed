import itertools
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import toolz
import zarr
from toolz import map

from cubed.backend_array_api import (
    backend_array_to_numpy_array,
    numpy_array_to_backend_array,
)
from cubed.runtime.types import CubedPipeline
from cubed.storage.zarr import T_ZarrArray, lazy_empty
from cubed.types import T_Chunks, T_DType, T_Shape, T_Store
from cubed.utils import chunk_memory, get_item, map_nested, split_into, to_chunksize
from cubed.vendor.dask.array.core import normalize_chunks
from cubed.vendor.dask.blockwise import _get_coord_mapping, _make_dims, lol_product
from cubed.vendor.dask.core import flatten

from .types import CubedArrayProxy, PrimitiveOperation

sym_counter = 0


def gensym(name: str) -> str:
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
        A function that maps an output chunk index to one or more input chunk indexes.
    function : Callable
        A function that maps input chunks to an output chunk.
    reads_map : Dict[str, CubedArrayProxy]
        Read proxy dictionary keyed by array name.
    write : CubedArrayProxy
        Write proxy with an ``array`` attribute that supports ``__setitem__``.
    """

    block_function: Callable[..., Any]
    function: Callable[..., Any]
    reads_map: Dict[str, CubedArrayProxy]
    write: CubedArrayProxy


def apply_blockwise(out_key: List[int], *, config: BlockwiseSpec) -> None:
    """Stage function for blockwise."""
    # lithops needs params to be lists not tuples, so convert back
    out_key_tuple = tuple(out_key)
    out_chunk_key = key_to_slices(
        out_key_tuple, config.write.array, config.write.chunks
    )

    # get array chunks for input keys, preserving any nested list structure
    args = []
    get_chunk_config = partial(get_chunk, config=config)
    name_chunk_inds = config.block_function(("out",) + out_key_tuple)
    for name_chunk_ind in name_chunk_inds:
        arg = map_nested(get_chunk_config, name_chunk_ind)
        args.append(arg)

    result = config.function(*args)
    if isinstance(result, dict):  # structured array with named fields
        for k, v in result.items():
            v = backend_array_to_numpy_array(v)
            config.write.open().set_basic_selection(out_chunk_key, v, fields=k)
    else:
        result = backend_array_to_numpy_array(result)
        config.write.open()[out_chunk_key] = result


def key_to_slices(
    key: Tuple[int, ...], arr: T_ZarrArray, chunks: Optional[T_Chunks] = None
) -> Tuple[slice, ...]:
    """Convert a chunk index key to a tuple of slices"""
    chunks = normalize_chunks(chunks or arr.chunks, shape=arr.shape, dtype=arr.dtype)
    return get_item(chunks, key)


def get_chunk(name_chunk_ind, config):
    """Read a chunk from the named array"""
    name = name_chunk_ind[0]
    chunk_ind = name_chunk_ind[1:]
    arr = config.reads_map[name].open()
    chunk_key = key_to_slices(chunk_ind, arr)
    arg = arr[chunk_key]
    arg = numpy_array_to_backend_array(arg)
    return arg


def blockwise(
    func: Callable[..., Any],
    out_ind: Sequence[Union[str, int]],
    *args: Any,
    allowed_mem: int,
    reserved_mem: int,
    target_store: T_Store,
    shape: T_Shape,
    dtype: T_DType,
    chunks: T_Chunks,
    new_axes: Optional[Dict[int, int]] = None,
    in_names: Optional[List[str]] = None,
    out_name: Optional[str] = None,
    extra_projected_mem: int = 0,
    extra_func_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Apply a function to multiple blocks from multiple inputs, expressed using concise indexing rules.

    Unlike ```general_blockwise``, an index notation is used to specify the block mapping,
    like in Dask Array.

    Parameters
    ----------
    func : callable
        Function to apply to individual tuples of blocks
    out_ind : iterable
        Block pattern of the output, something like 'ijk' or (1, 2, 3)
    *args : sequence of Array, index pairs
        Sequence like (x, 'ij', y, 'jk', z, 'i')
    allowed_mem : int
        The memory available to a worker for running a task, in bytes. Includes ``reserved_mem``.
    reserved_mem : int
        The memory reserved on a worker for non-data use when running a task, in bytes
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
    extra_projected_mem : int
        Extra memory projected to be needed (in bytes) in addition to the memory used reading
        the input arrays and writing the output.
    extra_func_kwargs : dict
        Extra keyword arguments to pass to function that can't be passed as regular keyword arguments
        since they clash with other blockwise arguments (such as dtype).
    **kwargs : dict
        Extra keyword arguments to pass to function

    Returns
    -------
    CubedPipeline to run the operation
    """

    arrays: Sequence[T_ZarrArray] = args[::2]
    array_names = in_names or [f"in_{i}" for i in range(len(arrays))]

    inds: Sequence[Union[str, int]] = args[1::2]

    numblocks: Dict[str, Tuple[int, ...]] = {}
    for name, array in zip(array_names, arrays):
        input_chunks = normalize_chunks(
            array.chunks, shape=array.shape, dtype=array.dtype
        )
        numblocks[name] = tuple(map(len, input_chunks))

    argindsstr: List[Any] = []
    for name, ind in zip(array_names, inds):
        argindsstr.extend((name, ind))

    block_function = make_blockwise_function_flattened(
        func,
        out_name or "out",
        out_ind,
        *argindsstr,
        numblocks=numblocks,
        new_axes=new_axes,
    )

    return general_blockwise(
        func,
        block_function,
        *arrays,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        target_store=target_store,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        in_names=in_names,
        extra_projected_mem=extra_projected_mem,
        extra_func_kwargs=extra_func_kwargs,
        **kwargs,
    )


def general_blockwise(
    func: Callable[..., Any],
    block_function: Callable[..., Any],
    *arrays: Any,
    allowed_mem: int,
    reserved_mem: int,
    target_store: T_Store,
    shape: T_Shape,
    dtype: T_DType,
    chunks: T_Chunks,
    in_names: Optional[List[str]] = None,
    extra_projected_mem: int = 0,
    extra_func_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """A more general form of ``blockwise`` that uses a function to specify the block
    mapping, rather than an index notation.

    Parameters
    ----------
    func : callable
        Function to apply to individual tuples of blocks
    block_function : callable
        A function that maps an output chunk index to one or more input chunk indexes.
    *arrays : sequence of Array
        The input arrays.
    allowed_mem : int
        The memory available to a worker for running a task, in bytes. Includes ``reserved_mem``.
    reserved_mem : int
        The memory reserved on a worker for non-data use when running a task, in bytes
    target_store : string or zarr.Array
        Path to output Zarr store, or Zarr array
    shape : tuple
        The shape of the output array.
    dtype : np.dtype
        The ``dtype`` of the output array.
    chunks : tuple
        The chunks of the output array.
    extra_projected_mem : int
        Extra memory projected to be needed (in bytes) in addition to the memory used reading
        the input arrays and writing the output.
    extra_func_kwargs : dict
        Extra keyword arguments to pass to function that can't be passed as regular keyword arguments
        since they clash with other blockwise arguments (such as dtype).
    **kwargs : dict
        Extra keyword arguments to pass to function

    Returns
    -------
    PrimitiveOperation to run the operation
    """
    array_names = in_names or [f"in_{i}" for i in range(len(arrays))]
    array_map = {name: array for name, array in zip(array_names, arrays)}

    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    chunksize = to_chunksize(chunks)
    if isinstance(target_store, zarr.Array):
        target_array = target_store
    else:
        target_array = lazy_empty(
            shape, dtype=dtype, chunks=chunksize, store=target_store
        )

    func_kwargs = extra_func_kwargs or {}
    func_with_kwargs = partial(func, **{**kwargs, **func_kwargs})
    read_proxies = {
        name: CubedArrayProxy(array, array.chunks) for name, array in array_map.items()
    }
    write_proxy = CubedArrayProxy(target_array, chunksize)
    spec = BlockwiseSpec(block_function, func_with_kwargs, read_proxies, write_proxy)

    # calculate projected memory
    projected_mem = reserved_mem + extra_projected_mem
    # inputs
    for array in arrays:  # inputs
        # memory for a compressed and an uncompressed input array chunk
        # - we assume compression has no effect (so it's an overestimate)
        # - ideally we'd be able to look at nbytes_stored,
        #   but this is not possible in general since the array has not been written yet
        projected_mem += chunk_memory(array.dtype, array.chunks) * 2
    # output
    # memory for a compressed and an uncompressed output array chunk
    # - this assumes the blockwise function creates a new array)
    # - numcodecs uses a working output buffer that's the size of the array being compressed
    projected_mem += chunk_memory(dtype, chunksize) * 2

    if projected_mem > allowed_mem:
        raise ValueError(
            f"Projected blockwise memory ({projected_mem}) exceeds allowed_mem ({allowed_mem}), including reserved_mem ({reserved_mem})"
        )

    # this must be an iterator of lists, not of tuples, otherwise lithops breaks
    output_blocks = map(list, itertools.product(*[range(len(c)) for c in chunks]))
    num_tasks = math.prod(len(c) for c in chunks)

    pipeline = CubedPipeline(
        apply_blockwise,
        gensym("apply_blockwise"),
        output_blocks,
        spec,
    )
    return PrimitiveOperation(
        pipeline=pipeline,
        target_array=target_array,
        projected_mem=projected_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
    )


# Code for fusing blockwise operations


def is_fuse_candidate(primitive_op: PrimitiveOperation) -> bool:
    """
    Return True if a primitive operation is a candidate for blockwise fusion.
    """
    return primitive_op.pipeline.function == apply_blockwise


def can_fuse_primitive_ops(
    primitive_op1: PrimitiveOperation, primitive_op2: PrimitiveOperation
) -> bool:
    if is_fuse_candidate(primitive_op1) and is_fuse_candidate(primitive_op2):
        return primitive_op1.num_tasks == primitive_op2.num_tasks
    return False


def can_fuse_multiple_primitive_ops(
    primitive_op: PrimitiveOperation, *predecessor_primitive_ops: PrimitiveOperation
) -> bool:
    if is_fuse_candidate(primitive_op) and all(
        is_fuse_candidate(p) for p in predecessor_primitive_ops
    ):
        return all(
            primitive_op.num_tasks == p.num_tasks for p in predecessor_primitive_ops
        )
    return False


def fuse(
    primitive_op1: PrimitiveOperation, primitive_op2: PrimitiveOperation
) -> PrimitiveOperation:
    """
    Fuse two blockwise operations into a single operation, avoiding writing to (or reading from) the target of the first operation.
    """

    assert primitive_op1.num_tasks == primitive_op2.num_tasks

    pipeline1 = primitive_op1.pipeline
    pipeline2 = primitive_op2.pipeline

    mappable = pipeline2.mappable

    def fused_blockwise_func(out_key):
        return pipeline1.config.block_function(
            *pipeline2.config.block_function(out_key)
        )

    def fused_func(*args):
        return pipeline2.config.function(pipeline1.config.function(*args))

    read_proxies = pipeline1.config.reads_map
    write_proxy = pipeline2.config.write
    spec = BlockwiseSpec(fused_blockwise_func, fused_func, read_proxies, write_proxy)

    target_array = primitive_op2.target_array
    projected_mem = max(primitive_op1.projected_mem, primitive_op2.projected_mem)
    reserved_mem = max(primitive_op1.reserved_mem, primitive_op2.reserved_mem)
    num_tasks = primitive_op2.num_tasks

    pipeline = CubedPipeline(
        apply_blockwise,
        gensym("fused_apply_blockwise"),
        mappable,
        spec,
    )
    return PrimitiveOperation(
        pipeline=pipeline,
        target_array=target_array,
        projected_mem=projected_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
    )


def fuse_multiple(
    primitive_op: PrimitiveOperation,
    *predecessor_primitive_ops: PrimitiveOperation,
    predecessor_funcs_nargs=None,
) -> PrimitiveOperation:
    """
    Fuse a blockwise operation and its predecessors into a single operation, avoiding writing to (or reading from) the targets of the predecessor operations.
    """

    assert all(
        primitive_op.num_tasks == p.num_tasks
        for p in predecessor_primitive_ops
        if p is not None
    )

    pipeline = primitive_op.pipeline
    predecessor_pipelines = [
        primitive_op.pipeline if primitive_op is not None else None
        for primitive_op in predecessor_primitive_ops
    ]

    mappable = pipeline.mappable

    def apply_pipeline_block_func(pipeline, arg):
        if pipeline is None:
            return (arg,)
        return pipeline.config.block_function(arg)

    def fused_blockwise_func(out_key):
        # this will change when multiple outputs are supported
        args = pipeline.config.block_function(out_key)
        # flatten one level of args as the fused_func adds back grouping structure
        func_args = tuple(
            item
            for p, a in zip(predecessor_pipelines, args)
            for item in apply_pipeline_block_func(p, a)
        )
        return func_args

    def apply_pipeline_func(pipeline, *args):
        if pipeline is None:
            return args[0]
        return pipeline.config.function(*args)

    def fused_func(*args):
        # split all args to the fused function into groups, one for each predecessor function
        split_args = split_into(args, predecessor_funcs_nargs)
        func_args = [
            apply_pipeline_func(p, *a)
            for p, a in zip(predecessor_pipelines, split_args)
        ]
        return pipeline.config.function(*func_args)

    read_proxies = dict(pipeline.config.reads_map)
    for p in predecessor_pipelines:
        if p is not None:
            read_proxies.update(p.config.reads_map)
    write_proxy = pipeline.config.write
    spec = BlockwiseSpec(fused_blockwise_func, fused_func, read_proxies, write_proxy)

    target_array = primitive_op.target_array
    projected_mem = max(
        primitive_op.projected_mem,
        *(p.projected_mem for p in predecessor_primitive_ops if p is not None),
    )
    reserved_mem = max(
        primitive_op.reserved_mem,
        *(p.reserved_mem for p in predecessor_primitive_ops if p is not None),
    )
    num_tasks = primitive_op.num_tasks

    fused_pipeline = CubedPipeline(
        apply_blockwise,
        gensym("fused_apply_blockwise"),
        mappable,
        spec,
    )
    return PrimitiveOperation(
        pipeline=fused_pipeline,
        target_array=target_array,
        projected_mem=projected_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
    )


# blockwise functions


def make_blockwise_function(
    func: Callable[..., Any],
    output: str,
    out_indices: Sequence[Union[str, int]],
    *arrind_pairs: Any,
    numblocks: Optional[Dict[str, Tuple[int, ...]]] = None,
    new_axes: Optional[Dict[int, int]] = None,
) -> Callable[[List[int]], Any]:
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
    func: Callable[..., Any],
    output: str,
    out_indices: Sequence[Union[str, int]],
    *arrind_pairs: Any,
    numblocks: Optional[Dict[str, Tuple[int, ...]]] = None,
    new_axes: Optional[Dict[int, int]] = None,
) -> Callable[[List[int]], Any]:
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
