import inspect
import itertools
import logging
import math
from collections.abc import Iterator
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
from cubed.primitive.memory import BufferCopies, MemoryModeller, calculate_projected_mem
from cubed.primitive.types import CubedArrayProxy, PrimitiveOperation
from cubed.runtime.types import CubedPipeline
from cubed.storage.zarr import LazyZarrArray, T_ZarrArray, lazy_zarr_array
from cubed.types import T_Chunks, T_DType, T_RegularChunks, T_Shape, T_Store
from cubed.utils import array_memory, chunk_memory, get_item, map_nested
from cubed.utils import numblocks as compute_numblocks
from cubed.utils import split_into, to_chunksize
from cubed.vendor.dask.array.core import normalize_chunks
from cubed.vendor.dask.blockwise import _get_coord_mapping, _make_dims, lol_product
from cubed.vendor.dask.core import flatten

logger = logging.getLogger(__name__)


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
    key_function : Callable
        A function that maps an output chunk key to one or more input chunk keys.
    function : Callable
        A function that maps input chunks to an output chunk.
    function_nargs: int
        The number of array arguments that ``function`` takes. Note that for some
        functions (``concat``, ``stack``) this may be different to the number of
        arrays that the operation depends on.
    num_input_blocks: Tuple[int, ...]
        The number of input blocks read from each input array.
    num_output_blocks: Tuple[int, ...]
        The number of output blocks written to each output array.
    iterable_input_blocks: Tuple[int, ...]
        Whether the input blocks read from each input array are supplied as an iterable or not.
    reads_map : Dict[str, CubedArrayProxy]
        Read proxy dictionary keyed by array name.
    writes_list : List[CubedArrayProxy]
        Write proxy list where entries have an ``array`` attribute that supports ``__setitem__``.
    """

    key_function: Callable[..., Any]
    function: Callable[..., Any]
    function_nargs: int
    num_input_blocks: Tuple[int, ...]
    num_output_blocks: Tuple[int, ...]
    iterable_input_blocks: Tuple[bool, ...]
    reads_map: Dict[str, CubedArrayProxy]
    writes_list: List[CubedArrayProxy]
    return_writes_stores: bool = False


def apply_blockwise(
    out_coords: List[int], *, config: BlockwiseSpec
) -> Optional[List[T_Store]]:
    """Stage function for blockwise."""
    # lithops needs params to be lists not tuples, so convert back
    out_coords_tuple = tuple(out_coords)

    results = get_results_in_different_scope(out_coords, config=config)

    # if blockwise function is a regular function (not a generator) that doesn't return multiple values then make it iterable
    if not inspect.isgeneratorfunction(config.function) and not isinstance(
        results, tuple
    ):
        results = (results,)
    for i, result in enumerate(results):
        out_chunk_key = key_to_slices(
            out_coords_tuple, config.writes_list[i].array, config.writes_list[i].chunks
        )
        if isinstance(result, dict):  # group of arrays with named fields
            for k, v in result.items():
                v = backend_array_to_numpy_array(v)
                config.writes_list[i].open().set_basic_selection(
                    out_chunk_key, v, fields=k
                )
        else:
            result = backend_array_to_numpy_array(result)
            config.writes_list[i].open()[out_chunk_key] = result

    if config.return_writes_stores:
        return [write_proxy.open().store for write_proxy in config.writes_list]
    return None


def get_results_in_different_scope(out_coords: List[int], *, config: BlockwiseSpec):
    # wrap function call in a function so that args go out of scope (and free memory) as soon as results are returned

    # lithops needs params to be lists not tuples, so convert back
    out_coords_tuple = tuple(out_coords)

    # get array chunks for input keys, preserving any nested list structure
    get_chunk_config = partial(get_chunk, config=config)
    out_key = ("out",) + out_coords_tuple  # array name is ignored by key_function
    name_chunk_inds = list(config.key_function(out_key))
    args = map_nested(get_chunk_config, name_chunk_inds)

    return config.function(*args)


def key_to_slices(
    key: Tuple[int, ...], arr: T_ZarrArray, chunks: Optional[T_Chunks] = None
) -> Tuple[slice, ...]:
    """Convert a chunk index key to a tuple of slices"""
    chunks = normalize_chunks(chunks or arr.chunks, shape=arr.shape, dtype=arr.dtype)
    return get_item(chunks, key)


def get_chunk(in_key, config):
    """Read a chunk from the named array"""
    name = in_key[0]
    in_coords = in_key[1:]
    arr = config.reads_map[name].open()
    selection = key_to_slices(in_coords, arr)
    arg = arr[selection]
    arg = numpy_array_to_backend_array(arg)
    return arg


def blockwise(
    func: Callable[..., Any],
    out_ind: Sequence[Union[str, int]],
    *args: Any,
    allowed_mem: int,
    reserved_mem: int,
    target_store: T_Store,
    target_path: Optional[str] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    compressor: Union[dict, str, None] = "default",
    shape: T_Shape,
    dtype: T_DType,
    chunks: T_Chunks,
    new_axes: Optional[Dict[int, int]] = None,
    in_names: Optional[List[str]] = None,
    out_name: Optional[str] = None,
    extra_projected_mem: int = 0,
    buffer_copies: Optional[BufferCopies] = None,
    extra_func_kwargs: Optional[Dict[str, Any]] = None,
    fusable_with_predecessors: bool = True,
    fusable_with_successors: bool = True,
    num_input_blocks: Optional[Tuple[int, ...]] = None,
    iterable_input_blocks: Optional[Tuple[bool, ...]] = None,
    **kwargs,
) -> PrimitiveOperation:
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
    buffer_copies: BufferCopies
        The the number of buffer copies incurred for array storage operations.
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
    for name, array in zip(array_names, arrays, strict=True):
        input_chunks = normalize_chunks(
            array.chunks, shape=array.shape, dtype=array.dtype
        )
        numblocks[name] = tuple(map(len, input_chunks))

    argindsstr: List[Any] = []
    for name, ind in zip(array_names, inds, strict=True):
        argindsstr.extend((name, ind))

    key_function = make_blockwise_key_function_flattened(
        func,
        out_name or "out",
        out_ind,
        *argindsstr,
        numblocks=numblocks,
        new_axes=new_axes,
    )

    return general_blockwise(
        func,
        key_function,
        *arrays,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        target_stores=[target_store],
        target_paths=[target_path] if target_path is not None else None,
        storage_options=storage_options,
        compressor=compressor,
        shapes=[shape],
        dtypes=[dtype],
        chunkss=[chunks],
        in_names=in_names,
        extra_projected_mem=extra_projected_mem,
        buffer_copies=buffer_copies,
        extra_func_kwargs=extra_func_kwargs,
        fusable_with_predecessors=fusable_with_predecessors,
        fusable_with_successors=fusable_with_successors,
        num_input_blocks=num_input_blocks,
        iterable_input_blocks=iterable_input_blocks,
        **kwargs,
    )


def general_blockwise(
    func: Callable[..., Any],
    key_function: Callable[..., Any],
    *arrays: Any,
    allowed_mem: int,
    reserved_mem: int,
    target_stores: List[T_Store],
    target_paths: Optional[List[str]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    compressor: Union[dict, str, None] = "default",
    shapes: List[T_Shape],
    dtypes: List[T_DType],
    chunkss: List[T_Chunks],
    in_names: Optional[List[str]] = None,
    extra_projected_mem: int = 0,
    buffer_copies: Optional[BufferCopies] = None,
    extra_func_kwargs: Optional[Dict[str, Any]] = None,
    fusable_with_predecessors: bool = True,
    fusable_with_successors: bool = True,
    function_nargs: Optional[int] = None,
    num_input_blocks: Optional[Tuple[int, ...]] = None,
    iterable_input_blocks: Optional[Tuple[bool, ...]] = None,
    target_chunks_: Optional[T_RegularChunks] = None,
    return_writes_stores: bool = False,
    **kwargs,
) -> PrimitiveOperation:
    """A more general form of ``blockwise`` that uses a function to specify the block
    mapping, rather than an index notation, and which supports multiple outputs.

    For multiple outputs, all output arrays must have matching numblocks.

    Parameters
    ----------
    func : callable
        Function to apply to individual tuples of blocks
    key_function : callable
        A function that maps an output chunk key to one or more input chunk keys.
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
    buffer_copies: BufferCopies
        The the number of buffer copies incurred for array storage operations.
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
    array_map = {name: array for name, array in zip(array_names, arrays, strict=True)}

    func_kwargs = extra_func_kwargs or {}
    func_with_kwargs = partial(func, **{**kwargs, **func_kwargs})
    function_nargs = function_nargs or len(arrays)
    num_input_blocks = num_input_blocks or (1,) * len(arrays)
    iterable_input_blocks = iterable_input_blocks or (False,) * len(arrays)
    read_proxies = {
        name: CubedArrayProxy(array, array.chunks) for name, array in array_map.items()
    }

    write_proxies = []
    output_chunk_memory = 0
    target_arrays = []

    numblocks0 = None
    for i, target_store in enumerate(target_stores):
        chunks_normal = normalize_chunks(chunkss[i], shape=shapes[i], dtype=dtypes[i])
        chunksize = to_chunksize(chunks_normal)
        if numblocks0 is None:
            numblocks0 = compute_numblocks(chunks_normal)
        else:
            numblocks = compute_numblocks(chunks_normal)
            if numblocks != numblocks0:
                raise ValueError(
                    f"All outputs must have matching number of blocks in each dimension. Chunks specified: {chunkss}"
                )
        ta: Union[zarr.Array, LazyZarrArray]
        if isinstance(target_store, zarr.Array):
            ta = target_store
        else:
            ta = lazy_zarr_array(
                target_store,
                shapes[i],
                dtype=dtypes[i],
                chunks=target_chunks_ or chunksize,
                path=target_paths[i] if target_paths is not None else None,
                storage_options=storage_options,
                compressor=compressor,
            )
        target_arrays.append(ta)

        write_proxies.append(CubedArrayProxy(ta, chunksize))

        # only one output chunk is read into memory at a time, so we find the largest
        output_chunk_memory = max(
            output_chunk_memory, array_memory(dtypes[i], chunksize)
        )

    # the number of blocks written to each target array is currently the same
    nb = math.prod(chunksize) // math.prod(target_chunks_ or chunksize)
    num_output_blocks = (nb,) * len(target_arrays)

    spec = BlockwiseSpec(
        key_function,
        func_with_kwargs,
        function_nargs,
        num_input_blocks,
        num_output_blocks,
        iterable_input_blocks,
        read_proxies,
        write_proxies,
        return_writes_stores,
    )

    buffer_copies = buffer_copies or BufferCopies(read=1, write=1)
    projected_mem = calculate_projected_mem(
        reserved_mem=reserved_mem,
        inputs=[array_memory(array.dtype, array.chunks) for array in arrays],
        operation=extra_projected_mem,
        output=output_chunk_memory,
        buffer_copies=buffer_copies,
    )

    if projected_mem > allowed_mem:
        raise ValueError(
            f"Projected blockwise memory ({projected_mem}) exceeds allowed_mem ({allowed_mem}), including reserved_mem ({reserved_mem})"
        )

    # this must be an iterator of lists, not of tuples, otherwise lithops breaks
    output_blocks = map(
        list, itertools.product(*[range(len(c)) for c in chunks_normal])
    )
    num_tasks = math.prod(len(c) for c in chunks_normal)

    pipeline = CubedPipeline(
        apply_blockwise,
        gensym("apply_blockwise"),
        output_blocks,
        spec,
    )
    return PrimitiveOperation(
        pipeline=pipeline,
        source_array_names=array_names,
        target_array=target_arrays[0] if len(target_arrays) == 1 else target_arrays,
        projected_mem=projected_mem,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
        fusable_with_predecessors=fusable_with_predecessors,
        fusable_with_successors=fusable_with_successors,
        write_chunks=chunksize,
    )


# Code for fusing blockwise operations


def is_fuse_candidate(primitive_op: PrimitiveOperation) -> bool:
    """
    Return True if a primitive operation is a candidate for blockwise fusion.
    """
    return primitive_op.pipeline.function == apply_blockwise and (
        primitive_op.fusable_with_predecessors or primitive_op.fusable_with_successors
    )


def can_fuse_primitive_ops(
    primitive_op1: PrimitiveOperation, primitive_op2: PrimitiveOperation
) -> bool:
    if is_fuse_candidate(primitive_op1) and is_fuse_candidate(primitive_op2):
        return primitive_op1.num_tasks == primitive_op2.num_tasks
    return False


def can_fuse_multiple_primitive_ops(
    name: str,
    primitive_op: PrimitiveOperation,
    predecessor_primitive_ops: List[PrimitiveOperation],
    *,
    max_total_num_input_blocks: Optional[int] = None,
) -> bool:
    if is_fuse_candidate(primitive_op) and all(
        p is None or is_fuse_candidate(p) for p in predecessor_primitive_ops
    ):
        # If the peak projected memory for running all the predecessor ops in
        # order is larger than allowed_mem then we can't fuse.
        peak_projected = peak_projected_mem(predecessor_primitive_ops)
        if peak_projected > primitive_op.allowed_mem:
            logger.debug(
                "can't fuse %s since peak projected memory for predecessor ops (%s) is greater than allowed (%s)",
                name,
                peak_projected,
                primitive_op.allowed_mem,
            )
            return False
        if max_total_num_input_blocks is None:
            # If max total input blocks not specified, then only fuse if num
            # tasks of predecessor ops match.
            ret = all(
                primitive_op.num_tasks == p.num_tasks
                for p in predecessor_primitive_ops
                if p is not None
            )
            if ret:
                logger.debug(
                    "can fuse %s since num tasks of predecessor ops match", name
                )
            else:
                logger.debug(
                    "can't fuse %s since num tasks of predecessor ops do not match",
                    name,
                )
            return ret
        else:
            num_input_blocks = primitive_op.pipeline.config.num_input_blocks
            total_num_input_blocks = 0
            for ni, p in zip(num_input_blocks, predecessor_primitive_ops, strict=True):
                if p is None:
                    continue
                for nj in p.pipeline.config.num_input_blocks:
                    total_num_input_blocks += ni * nj
            ret = total_num_input_blocks <= max_total_num_input_blocks
            if ret:
                logger.debug(
                    "can fuse %s since total number of input blocks (%s) does not exceed max (%s)",
                    name,
                    total_num_input_blocks,
                    max_total_num_input_blocks,
                )
            else:
                logger.debug(
                    "can't fuse %s since total number of input blocks (%s) exceeds max (%s)",
                    name,
                    total_num_input_blocks,
                    max_total_num_input_blocks,
                )
            return ret
    logger.debug(
        "can't fuse %s since primitive op and predecessors are not all candidates", name
    )
    return False


def peak_projected_mem(primitive_ops):
    """Calculate the peak projected memory for running a series of primitive ops
    and retaining their return values in memory."""
    memory_modeller = MemoryModeller()
    for p in primitive_ops:
        if p is None:
            continue
        memory_modeller.allocate(p.projected_mem)
        chunkmem = chunk_memory(p.target_array)
        memory_modeller.free(p.projected_mem - chunkmem)
    return memory_modeller.peak_mem


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

    def fused_key_func(out_key):
        return pipeline1.config.key_function(*pipeline2.config.key_function(out_key))

    def fused_func(*args):
        return pipeline2.config.function(pipeline1.config.function(*args))

    function_nargs = pipeline1.config.function_nargs
    read_proxies = pipeline1.config.reads_map
    write_proxies = pipeline2.config.writes_list
    return_writes_stores = pipeline2.config.return_writes_stores
    num_input_blocks = tuple(
        n * pipeline2.config.num_input_blocks[0]
        for n in pipeline1.config.num_input_blocks
    )
    num_output_blocks = pipeline2.config.num_output_blocks
    iterable_input_blocks = pipeline1.config.iterable_input_blocks
    spec = BlockwiseSpec(
        fused_key_func,
        fused_func,
        function_nargs,
        num_input_blocks,
        num_output_blocks,
        iterable_input_blocks,
        read_proxies,
        write_proxies,
        return_writes_stores,
    )

    source_array_names = primitive_op1.source_array_names
    target_array = primitive_op2.target_array
    projected_mem = max(primitive_op1.projected_mem, primitive_op2.projected_mem)
    allowed_mem = primitive_op2.allowed_mem
    reserved_mem = primitive_op2.reserved_mem
    num_tasks = primitive_op2.num_tasks

    pipeline = CubedPipeline(
        apply_blockwise,
        gensym("fused_apply_blockwise"),
        mappable,
        spec,
    )
    return PrimitiveOperation(
        pipeline=pipeline,
        source_array_names=source_array_names,
        target_array=target_array,
        projected_mem=projected_mem,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
        fusable_with_predecessors=True,
    )


def fuse_multiple(
    primitive_op: PrimitiveOperation, *predecessor_primitive_ops: PrimitiveOperation
) -> PrimitiveOperation:
    """
    Fuse a blockwise operation and its predecessors into a single operation, avoiding writing to (or reading from) the targets of the predecessor operations.
    """

    bw_spec = primitive_op.pipeline.config

    null_blockwise_spec = BlockwiseSpec(
        key_function=lambda x: (x,),
        function=lambda x: x,
        function_nargs=1,
        num_input_blocks=(1,),
        num_output_blocks=(1,),
        iterable_input_blocks=(False,),
        reads_map={},
        writes_list=[],
    )
    predecessor_bw_specs = [
        primitive_op.pipeline.config
        if primitive_op is not None
        else null_blockwise_spec
        for primitive_op in predecessor_primitive_ops
    ]

    spec = fuse_blockwise_specs(bw_spec, *predecessor_bw_specs)

    source_array_names = []
    for i, p in enumerate(predecessor_primitive_ops):
        if p is None:
            source_array_names.append(primitive_op.source_array_names[i])
        else:
            source_array_names.extend(p.source_array_names)
    target_array = primitive_op.target_array
    projected_mem = max(
        primitive_op.projected_mem,
        peak_projected_mem(p for p in predecessor_primitive_ops if p is not None),
    )
    allowed_mem = primitive_op.allowed_mem
    reserved_mem = primitive_op.reserved_mem
    num_tasks = primitive_op.num_tasks

    fused_pipeline = CubedPipeline(
        apply_blockwise,
        gensym("fused_apply_blockwise"),
        primitive_op.pipeline.mappable,
        spec,
    )
    return PrimitiveOperation(
        pipeline=fused_pipeline,
        source_array_names=source_array_names,
        target_array=target_array,
        projected_mem=projected_mem,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        num_tasks=num_tasks,
        fusable_with_predecessors=True,
    )


def fuse_blockwise_specs(
    bw_spec: BlockwiseSpec, *predecessor_bw_specs: BlockwiseSpec
) -> BlockwiseSpec:
    """
    Fuse a blockwise spec and its predecessors into a single spec.
    """

    predecessor_funcs_nargs = [bws.function_nargs for bws in predecessor_bw_specs]

    fused_key_func = make_fused_key_function(
        bw_spec.key_function,
        [bws.key_function for bws in predecessor_bw_specs],
        predecessor_funcs_nargs,
    )

    fused_func = make_fused_function(
        bw_spec.function,
        [bws.function for bws in predecessor_bw_specs],
        bw_spec.iterable_input_blocks,
    )

    fused_function_nargs = bw_spec.function_nargs
    num_input_blocks = bw_spec.num_input_blocks
    predecessor_num_blocks = [bws.num_input_blocks for bws in predecessor_bw_specs]
    # note that the length of fused_num_input_blocks is the same as the number of input arrays
    # but may be different to fused_function_nargs since the fused function groups args
    fused_num_input_blocks = tuple(
        itertools.chain(
            *(
                tuple(n * m for m in nb)
                for n, nb in zip(num_input_blocks, predecessor_num_blocks, strict=True)
            )
        )
    )
    fused_num_output_blocks = bw_spec.num_output_blocks
    predecessor_iterable_input_blocks = [
        bws.iterable_input_blocks for bws in predecessor_bw_specs
    ]
    fused_iterable_input_blocks = tuple(
        itertools.chain(*predecessor_iterable_input_blocks)
    )

    read_proxies = dict(bw_spec.reads_map)
    for bws in predecessor_bw_specs:
        read_proxies.update(bws.reads_map)
    write_proxies = bw_spec.writes_list
    return_writes_stores = bw_spec.return_writes_stores
    return BlockwiseSpec(
        fused_key_func,
        fused_func,
        fused_function_nargs,
        fused_num_input_blocks,
        fused_num_output_blocks,
        fused_iterable_input_blocks,
        read_proxies,
        write_proxies,
        return_writes_stores,
    )


def apply_blockwise_key_func(key_function, arg):
    if isinstance(arg, tuple):
        return key_function(arg)
    else:
        # more than one input block is being read from arg
        assert isinstance(arg, (list, Iterator))
        if isinstance(arg, list):
            return tuple(
                list(item) for item in zip(*(key_function(a) for a in arg), strict=True)
            )
        else:
            # Return iterators to avoid materializing all array blocks at
            # once.
            return tuple(
                iter(list(item))
                for item in zip(*(key_function(a) for a in arg), strict=True)
            )


def apply_blockwise_func(func, is_iterable, *args):
    if is_iterable is False:
        ret = func(*args)
    else:
        # More than one input block is being read from this group of args to primitive op.
        # Note that it is important that a list is not returned to avoid materializing all
        # array blocks at once.
        ret = map(lambda item: func(*item), zip(*args, strict=True))
    return ret


def make_fused_key_function(
    key_function, predecessor_key_functions, predecessor_funcs_nargs
):
    def fused_key_func(out_key):
        args = key_function(out_key)
        # split all args to the fused function into groups, one for each predecessor function
        func_args = tuple(
            item
            for pkf, a in zip(predecessor_key_functions, args, strict=True)
            for item in apply_blockwise_key_func(pkf, a)
        )
        return split_into(func_args, predecessor_funcs_nargs)

    return fused_key_func


def make_fused_function(function, predecessor_functions, iterable_input_blocks):
    def fused_func_single(*args):
        # args are grouped appropriately so they can be called by each predecessor function
        func_args = [
            apply_blockwise_func(pf, iterable_input_blocks[i], *a)
            for i, (pf, a) in enumerate(zip(predecessor_functions, args, strict=True))
        ]
        return function(*func_args)

    # multiple outputs
    def fused_func_generator(*args):
        # args are grouped appropriately so they can be called by each predecessor function
        func_args = [
            apply_blockwise_func(pf, iterable_input_blocks[i], *a)
            for i, (pf, a) in enumerate(zip(predecessor_functions, args, strict=True))
        ]
        yield from function(*func_args)

    return (
        fused_func_generator
        if inspect.isgeneratorfunction(function)
        else fused_func_single
    )


# blockwise key functions


def make_blockwise_key_function(
    func: Callable[..., Any],
    output: str,
    out_indices: Sequence[Union[str, int]],
    *arrind_pairs: Any,
    numblocks: Dict[str, Tuple[int, ...]],
    new_axes: Optional[Dict[int, int]] = None,
) -> Callable[[List[int]], Any]:
    """Make a function that is the equivalent of make_blockwise_graph."""

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

    for axes, (arg, _) in zip(concat_axes, argpairs, strict=True):
        for ax in axes:
            if numblocks[arg][ax] > 1:
                raise ValueError(
                    f"Cannot have multiple chunks in dropped axis {ax}. "
                    "To fix, use a reduction after calling map_blocks "
                    "without specifying drop_axis, or rechunk first."
                )

    def key_function(out_key):
        out_coords = out_key[1:]

        # from Dask make_blockwise_graph
        deps = set()
        coords = out_coords + dummies
        args = []
        for cmap, axes, (arg, ind) in zip(
            coord_maps, concat_axes, argpairs, strict=True
        ):
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

    return key_function


def make_blockwise_key_function_flattened(
    func: Callable[..., Any],
    output: str,
    out_indices: Sequence[Union[str, int]],
    *arrind_pairs: Any,
    numblocks: Dict[str, Tuple[int, ...]],
    new_axes: Optional[Dict[int, int]] = None,
) -> Callable[[List[int]], Any]:
    # TODO: make this a part of make_blockwise_key_function?
    key_function = make_blockwise_key_function(
        func, output, out_indices, *arrind_pairs, numblocks=numblocks, new_axes=new_axes
    )

    def blockwise_fn_flattened(out_key):
        in_keys = key_function(out_key)[1:]  # drop function in position 0
        # flatten (nested) lists indicating contraction
        if isinstance(in_keys[0], list):
            in_keys = list(flatten(in_keys))
        return in_keys

    return blockwise_fn_flattened
