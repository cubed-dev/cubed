import builtins
import math
import numbers
from dataclasses import dataclass
from functools import partial
from itertools import product
from numbers import Integral, Number
from operator import add
from typing import TYPE_CHECKING, Any, Sequence, Tuple, Union
from warnings import warn

import ndindex
import numpy as np
import zarr
from tlz import concat, first, partition
from toolz import accumulate, map

from cubed import config
from cubed.backend_array_api import backend_array_to_numpy_array
from cubed.backend_array_api import namespace as nxp
from cubed.backend_array_api import numpy_array_to_backend_array
from cubed.core.array import CoreArray, check_array_specs, compute, gensym
from cubed.core.plan import Plan, new_temp_path
from cubed.primitive.blockwise import blockwise as primitive_blockwise
from cubed.primitive.blockwise import general_blockwise as primitive_general_blockwise
from cubed.primitive.memory import get_buffer_copies
from cubed.primitive.rechunk import rechunk as primitive_rechunk
from cubed.spec import spec_from_config
from cubed.storage.backend import open_backend_array
from cubed.types import T_RegularChunks, T_Shape
from cubed.utils import array_memory, array_size, get_item
from cubed.utils import numblocks as compute_numblocks
from cubed.utils import offset_to_block_id, to_chunksize
from cubed.vendor.dask.array.core import normalize_chunks
from cubed.vendor.dask.array.utils import validate_axis
from cubed.vendor.dask.blockwise import broadcast_dimensions
from cubed.vendor.dask.utils import has_keyword
from cubed.vendor.rechunker.algorithm import multistage_rechunking_plan

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array


def from_array(x, chunks="auto", asarray=None, spec=None) -> "Array":
    """Create a Cubed array from an array-like object."""

    if isinstance(x, CoreArray):
        raise ValueError(
            "Array is already a Cubed array. Use 'asarray' or 'rechunk' instead."
        )

    previous_chunks = getattr(x, "chunks", None)
    outchunks = normalize_chunks(
        chunks, x.shape, dtype=x.dtype, previous_chunks=previous_chunks
    )

    if isinstance(x, zarr.Array):  # zarr fast path
        from cubed.array_api import Array

        name = gensym()
        target = x
        plan = Plan._new(name, "from_array", target)
        arr = Array(name, target, spec, plan)

        chunksize = to_chunksize(outchunks)
        if chunks != "auto" and previous_chunks != chunksize:
            arr = rechunk(arr, chunksize)
        return arr

    if asarray is None:
        asarray = not hasattr(x, "__array_function__")

    return map_blocks(
        _from_array,
        dtype=x.dtype,
        chunks=outchunks,
        spec=spec,
        input_array=x,
        outchunks=outchunks,
        asarray=asarray,
    )


def _from_array(block, input_array, outchunks=None, asarray=None, block_id=None):
    out = input_array[get_item(outchunks, block_id)]
    if asarray:
        out = np.asarray(out)
    out = numpy_array_to_backend_array(out)
    return out


def from_zarr(store, path=None, spec=None) -> "Array":
    """Load an array from Zarr storage.

    Parameters
    ----------
    store : string or Zarr Store
        Input Zarr store
    path : string, optional
        Group path
    spec : cubed.Spec, optional
        The spec to use for the computation.

    Returns
    -------
    cubed.Array
        The array loaded from Zarr storage.
    """
    name = gensym()
    spec = spec or spec_from_config(config)
    target = open_backend_array(
        store,
        mode="r",
        path=path,
        storage_options=spec.storage_options,
    )

    from cubed.array_api import Array

    plan = Plan._new(name, "from_zarr", target)
    return Array(name, target, spec, plan)


def store(sources: Union["Array", Sequence["Array"]], targets, executor=None, **kwargs):
    """Save source arrays to array-like objects.

    In the current implementation ``targets`` must be Zarr arrays.

    Note that this operation is eager, and will run the computation
    immediately.

    Parameters
    ----------
    x : cubed.Array or collection of cubed.Array
        Arrays to save
    store : zarr.Array or collection of zarr.Array
        Zarr arrays to write to
    executor : cubed.runtime.types.Executor, optional
        The executor to use to run the computation.
        Defaults to using the in-process Python executor.
    """
    if isinstance(sources, CoreArray):
        sources = [sources]
        targets = [targets]

    if any(not isinstance(s, CoreArray) for s in sources):
        raise ValueError("All sources must be cubed array objects")

    if len(sources) != len(targets):
        raise ValueError(
            f"Different number of sources ({len(sources)}) and targets ({len(targets)})"
        )

    arrays = []
    for source, target in zip(sources, targets):
        identity = lambda a: a
        ind = tuple(range(source.ndim))
        array = blockwise(
            identity,
            ind,
            source,
            ind,
            dtype=source.dtype,
            align_arrays=False,
            target_store=target,
        )
        arrays.append(array)
    compute(*arrays, executor=executor, _return_in_memory_array=False, **kwargs)


def to_zarr(x: "Array", store, path=None, executor=None, **kwargs):
    """Save an array to Zarr storage.

    Note that this operation is eager, and will run the computation
    immediately.

    Parameters
    ----------
    x : cubed.Array
        Array to save
    store : string or Zarr Store
        Output Zarr store
    path : string, optional
        Group path
    executor : cubed.runtime.types.Executor, optional
        The executor to use to run the computation.
        Defaults to using the in-process Python executor.
    """
    # Note that the intermediate write to x's store will be optimized away
    # by map fusion (if it was produced with a blockwise operation).
    identity = lambda a: a
    ind = tuple(range(x.ndim))
    out = blockwise(
        identity,
        ind,
        x,
        ind,
        dtype=x.dtype,
        align_arrays=False,
        target_store=store,
        target_path=path,
    )
    out.compute(executor=executor, _return_in_memory_array=False, **kwargs)


def blockwise(
    func,
    out_ind,
    *args: Any,  # can't type this as mypy assumes args are all same type, but blockwise args alternate types
    dtype=None,
    adjust_chunks=None,
    new_axes=None,
    align_arrays=True,
    target_store=None,
    target_path=None,
    extra_func_kwargs=None,
    **kwargs,
) -> "Array":
    arrays = args[::2]

    assert len(arrays) > 0

    # Calculate output chunking. A lot of this is from dask's blockwise function

    new_axes = new_axes or {}
    new = (
        set(out_ind)
        - {a for arg in args[1::2] if arg is not None for a in arg}
        - set(new_axes or ())
    )
    if new:
        raise ValueError("Unknown dimension", new)

    if align_arrays:
        chunkss, arrays = unify_chunks(*args)
    else:
        inds = args[1::2]
        arginds = zip(arrays, inds)

        chunkss = {}
        # For each dimension, use the input chunking that has the most blocks;
        # this will ensure that broadcasting works as expected, and in
        # particular the number of blocks should be correct if the inputs are
        # consistent.
        for arg, ind in arginds:
            arg_chunks = normalize_chunks(
                arg.chunks, shape=arg.shape, dtype=arg.dtype
            )  # have to normalize zarr chunks
            for c, i in zip(arg_chunks, ind):
                if i not in chunkss or len(c) > len(chunkss[i]):
                    chunkss[i] = c

    for k, v in new_axes.items():
        if not isinstance(v, tuple):
            v = (v,)
        chunkss[k] = v

    chunks = [chunkss[i] for i in out_ind]
    if adjust_chunks:
        for i, ind in enumerate(out_ind):
            if ind in adjust_chunks:
                if callable(adjust_chunks[ind]):
                    chunks[i] = tuple(map(adjust_chunks[ind], chunks[i]))
                elif isinstance(adjust_chunks[ind], numbers.Integral):
                    chunks[i] = tuple(adjust_chunks[ind] for _ in chunks[i])
                elif isinstance(adjust_chunks[ind], (tuple, list)):
                    if len(adjust_chunks[ind]) != len(chunks[i]):
                        raise ValueError(
                            f"Dimension {i} has {len(chunks[i])} blocks, adjust_chunks "
                            f"specified with {len(adjust_chunks[ind])} blocks"
                        )
                    chunks[i] = tuple(adjust_chunks[ind])
                else:
                    raise NotImplementedError(
                        "adjust_chunks values must be callable, int, or tuple"
                    )
    _chunks = tuple(chunks)
    shape = tuple(map(sum, _chunks))

    # replace arrays with zarr arrays
    zargs = list(args)
    zargs[::2] = [a.zarray_maybe_lazy for a in arrays]
    in_names = [a.name for a in arrays]

    extra_source_arrays = kwargs.pop("extra_source_arrays", [])
    source_arrays = list(arrays) + list(extra_source_arrays)

    extra_projected_mem = kwargs.pop("extra_projected_mem", 0)

    fusable_with_predecessors = kwargs.pop("fusable_with_predecessors", True)
    fusable_with_successors = kwargs.pop("fusable_with_successors", True)
    num_input_blocks = kwargs.pop("num_input_blocks", None)
    iterable_input_blocks = kwargs.pop("iterable_input_blocks", None)

    name = gensym()
    spec = check_array_specs(arrays)
    buffer_copies = get_buffer_copies(spec)
    if target_store is None:
        target_store = new_temp_path(name=name, spec=spec)
    op = primitive_blockwise(
        func,
        out_ind,
        *zargs,
        allowed_mem=spec.allowed_mem,
        reserved_mem=spec.reserved_mem,
        extra_projected_mem=extra_projected_mem,
        target_store=target_store,
        target_path=target_path,
        storage_options=spec.storage_options,
        compressor=spec.zarr_compressor,
        shape=shape,
        dtype=dtype,
        chunks=_chunks,
        new_axes=new_axes,
        in_names=in_names,
        out_name=name,
        buffer_copies=buffer_copies,
        extra_func_kwargs=extra_func_kwargs,
        fusable_with_predecessors=fusable_with_predecessors,
        fusable_with_successors=fusable_with_successors,
        num_input_blocks=num_input_blocks,
        iterable_input_blocks=iterable_input_blocks,
        **kwargs,
    )
    plan = Plan._new(
        name,
        "blockwise",
        op.target_array,
        op,
        False,
        *source_arrays,
    )
    from cubed.array_api import Array

    return Array(name, op.target_array, spec, plan)


def general_blockwise(
    func,
    key_function,
    *arrays,
    shapes,
    dtypes,
    chunkss,
    target_stores=None,
    target_paths=None,
    extra_func_kwargs=None,
    **kwargs,
) -> Union["Array", Tuple["Array", ...]]:
    if has_keyword(func, "block_id"):
        from cubed.array_api.creation_functions import offsets_virtual_array

        # Create an array of index offsets with the same chunk structure as the args,
        # which we convert to block ids (chunk coordinates) later.
        array0 = arrays[0]
        # note that primitive general_blockwise checks that all chunkss have same numblocks
        numblocks = compute_numblocks(chunkss[0])
        offsets = offsets_virtual_array(numblocks, array0.spec)
        new_arrays = arrays + (offsets,)

        def key_function_with_offset(key_function):
            def wrap(out_key):
                out_coords = out_key[1:]
                offset_in_key = ((offsets.name,) + out_coords,)
                return key_function(out_key) + offset_in_key

            return wrap

        def func_with_block_id(func):
            def wrap(*a, **kw):
                offset = int(a[-1])  # convert from 0-d array
                block_id = offset_to_block_id(offset, numblocks)
                return func(*a[:-1], block_id=block_id, **kw)

            return wrap

        function_nargs = kwargs.pop("function_nargs", None)
        if function_nargs is not None:
            function_nargs = function_nargs + 1  # for offsets array
        num_input_blocks = kwargs.pop("num_input_blocks", None)
        if num_input_blocks is not None:
            num_input_blocks = num_input_blocks + (1,)  # for offsets array
        iterable_input_blocks = kwargs.pop("iterable_input_blocks", None)
        if iterable_input_blocks is not None:
            iterable_input_blocks = iterable_input_blocks + (
                False,
            )  # for offsets array

        return _general_blockwise(
            func_with_block_id(func),
            key_function_with_offset(key_function),
            *new_arrays,
            shapes=shapes,
            dtypes=dtypes,
            chunkss=chunkss,
            target_stores=target_stores,
            target_paths=target_paths,
            extra_func_kwargs=extra_func_kwargs,
            function_nargs=function_nargs,
            num_input_blocks=num_input_blocks,
            iterable_input_blocks=iterable_input_blocks,
            **kwargs,
        )

    return _general_blockwise(
        func,
        key_function,
        *arrays,
        shapes=shapes,
        dtypes=dtypes,
        chunkss=chunkss,
        target_stores=target_stores,
        target_paths=target_paths,
        extra_func_kwargs=extra_func_kwargs,
        **kwargs,
    )


def _general_blockwise(
    func,
    key_function,
    *arrays,
    shapes,
    dtypes,
    chunkss,
    target_stores=None,
    target_paths=None,
    extra_func_kwargs=None,
    **kwargs,
) -> Union["Array", Tuple["Array", ...]]:
    assert len(arrays) > 0

    # replace arrays with zarr arrays
    zargs = [a.zarray_maybe_lazy for a in arrays]
    in_names = [a.name for a in arrays]

    extra_source_arrays = kwargs.pop("extra_source_arrays", [])
    source_arrays = list(arrays) + list(extra_source_arrays)

    extra_projected_mem = kwargs.pop("extra_projected_mem", 0)

    num_input_blocks = kwargs.pop("num_input_blocks", None)
    iterable_input_blocks = kwargs.pop("iterable_input_blocks", None)

    op_name = kwargs.pop("op_name", "blockwise")

    spec = check_array_specs(arrays)
    buffer_copies = get_buffer_copies(spec)

    if isinstance(target_stores, list):  # multiple outputs
        name = [gensym() for _ in range(len(target_stores))]
        target_stores = [
            ts if ts is not None else new_temp_path(name=n, spec=spec)
            for n, ts in zip(name, target_stores)
        ]
    else:  # single output
        name = gensym()
        if target_stores is None:
            target_stores = [new_temp_path(name=name, spec=spec)]

    op = primitive_general_blockwise(
        func,
        key_function,
        *zargs,
        allowed_mem=spec.allowed_mem,
        reserved_mem=spec.reserved_mem,
        extra_projected_mem=extra_projected_mem,
        buffer_copies=buffer_copies,
        target_stores=target_stores,
        target_paths=target_paths,
        storage_options=spec.storage_options,
        compressor=spec.zarr_compressor,
        shapes=shapes,
        dtypes=dtypes,
        chunkss=chunkss,
        in_names=in_names,
        extra_func_kwargs=extra_func_kwargs,
        num_input_blocks=num_input_blocks,
        iterable_input_blocks=iterable_input_blocks,
        **kwargs,
    )
    plan = Plan._new(
        name,
        op_name,
        op.target_array,
        op,
        False,
        *source_arrays,
    )
    from cubed.array_api import Array

    if isinstance(op.target_array, list):  # multiple outputs
        return tuple(Array(n, ta, spec, plan) for n, ta in zip(name, op.target_array))
    else:  # single output
        return Array(name, op.target_array, spec, plan)


def elemwise(func, *args: "Array", dtype=None) -> "Array":
    """Apply a function elementwise to array arguments, respecting broadcasting."""
    shapes = [arg.shape for arg in args]
    out_ndim = len(np.broadcast_shapes(*shapes))
    expr_inds = tuple(range(out_ndim))[::-1]
    if dtype is None:
        raise ValueError("dtype must be specified for elemwise")
    return blockwise(
        func,
        expr_inds,
        *concat((a, tuple(range(a.ndim)[::-1])) for a in args),
        dtype=dtype,
    )


def index(x, key):
    "Subset an array, along one or more axes."
    if not isinstance(key, tuple):
        key = (key,)

    # Replace Cubed arrays with NumPy arrays - note that this may trigger a computation!
    # Note that NumPy arrays are needed for ndindex.
    key = tuple(
        backend_array_to_numpy_array(dim_sel.compute())
        if isinstance(dim_sel, CoreArray)
        else dim_sel
        for dim_sel in key
    )

    # Canonicalize index
    idx = ndindex.ndindex(key)
    idx = idx.expand(x.shape)

    # Remove newaxis values, to be filled in with expand_dims at end
    where_newaxis = [
        i for i, ia in enumerate(idx.args) if isinstance(ia, ndindex.Newaxis)
    ]
    for i, a in enumerate(where_newaxis):
        n = sum(isinstance(ia, ndindex.Integer) for ia in idx.args[:a])
        if n:
            where_newaxis[i] -= n
    idx = ndindex.Tuple(*(ia for ia in idx.args if not isinstance(ia, ndindex.Newaxis)))
    selection = idx.raw

    # Check selection is supported
    if any(ia.step < 1 for ia in idx.args if isinstance(ia, ndindex.Slice)):
        raise NotImplementedError(f"Slice step must be >= 1: {key}")
    if not all(
        isinstance(ia, (ndindex.Integer, ndindex.Slice, ndindex.IntegerArray))
        for ia in idx.args
    ):
        raise NotImplementedError(
            "Only integer, slice, or integer array indexes are allowed."
        )
    if sum(1 for ia in idx.args if isinstance(ia, ndindex.IntegerArray)) > 1:
        raise NotImplementedError("Only one integer array index is allowed.")

    # Use ndindex to find the resulting array shape and chunks

    def chunk_len_for_indexer(ia, c):
        if not isinstance(ia, ndindex.Slice):
            return c
        return max(c // ia.step, 1)

    def merged_chunk_len_for_indexer(ia, c):
        if not isinstance(ia, ndindex.Slice):
            return c
        if ia.step == 1:
            return c
        if (c // ia.step) < 1:
            return c
        # note that this may not be the same as c
        # but it is guaranteed to be a multiple of the corresponding
        # value returned by chunk_len_for_indexer, which is required
        # by merge_chunks
        return (c // ia.step) * ia.step

    shape = idx.newshape(x.shape)

    if shape == x.shape:
        # no op case (except possibly newaxis applied below)
        out = x
    elif array_size(shape) == 0:
        # empty output case
        from cubed.array_api.creation_functions import empty

        out = empty(shape, dtype=x.dtype, spec=x.spec)
    else:
        dtype = x.dtype
        chunks = tuple(
            chunk_len_for_indexer(ia, c)
            for ia, c in zip(idx.args, x.chunksize)
            if not isinstance(ia, ndindex.Integer)
        )

        # this is the same as chunks, except it has the same number of dimensions as the input
        out_chunksizes = tuple(
            chunk_len_for_indexer(ia, c) if not isinstance(ia, ndindex.Integer) else 1
            for ia, c in zip(idx.args, x.chunksize)
        )

        target_chunks = normalize_chunks(chunks, shape, dtype=dtype)

        # use map_selection (which uses general_blockwise) to allow more opportunities for optimization than map_direct

        def selection_function(out_key):
            out_coords = out_key[1:]
            return _target_chunk_selection(target_chunks, out_coords, selection)

        max_num_input_blocks = _index_num_input_blocks(
            idx, x.chunksize, out_chunksizes, x.numblocks
        )

        out = map_selection(
            None,  # no function to apply after selection
            selection_function,
            x,
            shape,
            x.dtype,
            target_chunks,
            max_num_input_blocks=max_num_input_blocks,
        )

        # merge chunks for any dims with step > 1 so they are
        # the same size as the input (or slightly smaller due to rounding)
        merged_chunks = tuple(
            merged_chunk_len_for_indexer(ia, c)
            for ia, c in zip(idx.args, x.chunksize)
            if not isinstance(ia, ndindex.Integer)
        )
        if chunks != merged_chunks:
            out = merge_chunks(out, merged_chunks)

    for axis in where_newaxis:
        from cubed.array_api.manipulation_functions import expand_dims

        out = expand_dims(out, axis=axis)

    return out


def _index_num_input_blocks(
    idx: ndindex.Tuple, in_chunksizes, out_chunksizes, numblocks
):
    num = 1
    for ia, c, oc, nb in zip(idx.args, in_chunksizes, out_chunksizes, numblocks):
        if isinstance(ia, ndindex.Integer) or nb == 1:
            pass  # single block
        elif isinstance(ia, ndindex.Slice):
            if (ia.start // c) == ((ia.stop - 1) // c):
                pass  # within same block
            elif ia.start % c != 0:
                num *= 2  # doesn't start on chunk boundary
            elif ia.step is not None and c % ia.step != 0 and oc > 1:
                # step is not a multiple of chunk size, and output chunks have more than one element
                # so some output chunks will access two input chunks
                num *= 2
        elif isinstance(ia, ndindex.IntegerArray):
            # in the worse case, elements could be retrieved from all blocks
            # TODO: improve to calculate the actual max input blocks
            num *= nb
        else:
            raise NotImplementedError(
                "Only integer, slice, or int array indexes are supported."
            )
    return num


def _create_zarr_indexer(selection, shape, chunks):
    if zarr.__version__[0] == "3":
        from zarr.core.chunk_grids import RegularChunkGrid
        from zarr.core.indexing import OrthogonalIndexer

        return OrthogonalIndexer(selection, shape, RegularChunkGrid(chunk_shape=chunks))
    else:
        from zarr.indexing import OrthogonalIndexer

        return OrthogonalIndexer(selection, ZarrArrayIndexingAdaptor(shape, chunks))


@dataclass
class ZarrArrayIndexingAdaptor:
    _shape: T_Shape
    _chunks: T_RegularChunks

    @classmethod
    def from_zarr_array(cls, zarray):
        return cls(zarray.shape, zarray.chunks)


def _assemble_index_chunk(
    arrays,
    dtype=None,
    func=None,
    selection_function=None,
    in_shape=None,
    in_chunksize=None,
    block_id=None,
    **kwargs,
):
    assert not isinstance(
        arrays, list
    ), "index expects an iterator of array blocks, not a list"

    # compute the selection on x required to get the relevant chunk for out_coords
    out_coords = block_id
    in_sel = selection_function(("out",) + out_coords)

    # use a Zarr indexer to convert this to input coordinates
    indexer = _create_zarr_indexer(in_sel, in_shape, in_chunksize)

    shape = indexer.shape
    out = np.empty(shape, dtype=dtype)

    if array_size(shape) > 0:
        _, lchunk_selection, lout_selection, *_ = zip(*indexer)
        for ai, chunk_select, out_select in zip(
            arrays, lchunk_selection, lout_selection
        ):
            out[out_select] = ai[chunk_select]

    if func is not None:
        if has_keyword(func, "block_id"):
            out = func(out, block_id=block_id, **kwargs)
        else:
            out = func(out, **kwargs)
    return out


def _target_chunk_selection(target_chunks, idx, selection):
    # integer, integer array, and slice indexes can be interspersed in selection
    # idx is the chunk index for the output (target_chunks)

    sel = []
    i = 0  # index into target_chunks and idx
    for s in selection:
        if isinstance(s, slice):
            offset = s.start or 0
            step = s.step if s.step is not None else 1
            start = tuple(
                accumulate(add, tuple(x * step for x in target_chunks[i]), offset)
            )
            j = idx[i]
            sel.append(slice(start[j], start[j + 1], step))
            i += 1
        # ndindex uses np.ndarray for integer arrays
        elif isinstance(s, np.ndarray):
            # find the cumulative chunk starts
            target_chunk_starts = [0] + list(
                accumulate(add, [c for c in target_chunks[i]])
            )
            # and use to slice the integer array
            j = idx[i]
            sel.append(s[target_chunk_starts[j] : target_chunk_starts[j + 1]])
            i += 1
        elif isinstance(s, int):
            sel.append(s)
            # don't increment i since integer indexes don't have a dimension in the target
        else:
            raise ValueError(f"Unsupported selection: {s}")
    return tuple(sel)


def map_selection(
    func,
    selection_function,
    x,
    shape,
    dtype,
    chunks,
    max_num_input_blocks,
    **kwargs,
) -> "Array":
    """
    Apply a function to selected subsets of an input array using standard NumPy indexing notation.

    Parameters
    ----------
    func : callable
        Function to apply to every block to produce the output array.
        Must accept ``block_id`` as a keyword argument (with same meaning as for ``map_blocks``).
    selection_function : callable
        A function that maps an output chunk key to one or more selections on the input array.
    x: Array
        The input array.
    shape : tuple
        Shape of the output array.
    dtype : np.dtype
        The ``dtype`` of the output array.
    chunks : tuple
        Chunk shape of blocks in the output array.
    max_num_input_blocks : int
        The maximum number of input blocks read from the input array.
    """

    def key_function(out_key):
        # compute the selection on x required to get the relevant chunk for out_key
        in_sel = selection_function(out_key)

        # use a Zarr indexer to convert selection to input coordinates
        indexer = _create_zarr_indexer(in_sel, x.shape, x.chunksize)

        return (iter(tuple((x.name,) + cp.chunk_coords for cp in indexer)),)

    num_input_blocks = (max_num_input_blocks,)
    iterable_input_blocks = (True,)

    out = general_blockwise(
        _assemble_index_chunk,
        key_function,
        x,
        shapes=[shape],
        dtypes=[dtype],
        chunkss=[chunks],
        extra_func_kwargs=dict(func=func, dtype=x.dtype),
        num_input_blocks=num_input_blocks,
        iterable_input_blocks=iterable_input_blocks,
        selection_function=selection_function,
        in_shape=x.shape,
        in_chunksize=x.chunksize,
        **kwargs,
    )
    from cubed import Array

    assert isinstance(out, Array)  # single output
    return out


def map_blocks(
    func,
    *args: "Array",
    dtype=None,
    chunks=None,
    drop_axis=None,
    new_axis=None,
    spec=None,
    **kwargs,
) -> "Array":
    """Apply a function to corresponding blocks from multiple input arrays."""

    from cubed.array_api.creation_functions import asarray

    # Coerce all args to Cubed arrays
    specs = [a.spec for a in args if hasattr(a, "spec")]
    spec0 = specs[0] if len(specs) > 0 else spec
    args = tuple(asarray(a, spec=spec0) for a in args)

    # Handle the case where an array is created by calling `map_blocks` with no input arrays
    if len(args) == 0:
        from cubed.array_api.creation_functions import empty_virtual_array

        shape = tuple(map(sum, chunks))
        args = (empty_virtual_array(shape, dtype=dtype, chunks=chunks, spec=spec),)

    if has_keyword(func, "block_id"):
        from cubed.array_api.creation_functions import offsets_virtual_array

        # Create an array of index offsets with the same chunk structure as the args,
        # which we convert to block ids (chunk coordinates) later.
        arg0 = args[0]
        numblocks = arg0.numblocks
        offsets = offsets_virtual_array(numblocks, arg0.spec)
        new_args = args + (offsets,)

        def func_with_block_id(func):
            def wrap(*a, **kw):
                offset = int(a[-1])  # convert from 0-d array
                block_id = offset_to_block_id(offset, numblocks)
                return func(*a[:-1], block_id=block_id, **kw)

            return wrap

        return _map_blocks(
            func_with_block_id(func),
            *new_args,
            dtype=dtype,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            **kwargs,
        )

    return _map_blocks(
        func,
        *args,
        dtype=dtype,
        chunks=chunks,
        drop_axis=drop_axis,
        new_axis=new_axis,
        **kwargs,
    )


def _map_blocks(
    func,
    *args: "Array",
    dtype=None,
    chunks=None,
    drop_axis=None,
    new_axis=None,
    **kwargs,
) -> "Array":
    # based on dask

    if drop_axis is None:
        drop_axis = []

    new_axes = {}

    if isinstance(drop_axis, Number):
        drop_axis = [drop_axis]
    if isinstance(new_axis, Number):
        new_axis = [new_axis]

    arrs = args
    argpairs = [
        (a, tuple(range(a.ndim))[::-1]) if isinstance(a, CoreArray) else (a, None)
        for a in args
    ]
    if arrs:
        out_ind = tuple(range(max(a.ndim for a in arrs)))[::-1]
    else:
        out_ind = ()

    if drop_axis:
        ndim_out = len(out_ind)
        if any(i < -ndim_out or i >= ndim_out for i in drop_axis):
            raise ValueError(
                f"drop_axis out of range (drop_axis={drop_axis}, "
                f"but output is {ndim_out}d)."
            )
        drop_axis = [i % ndim_out for i in drop_axis]
        out_ind = tuple(x for i, x in enumerate(out_ind) if i not in drop_axis)
    if new_axis is None and chunks is not None and len(out_ind) < len(chunks):
        new_axis = range(len(chunks) - len(out_ind))
    if new_axis:
        # new_axis = [x + len(drop_axis) for x in new_axis]
        temp_out_ind = list(out_ind)
        for ax in sorted(new_axis):
            n = len(temp_out_ind) + len(drop_axis)
            temp_out_ind.insert(ax, n)
            if chunks is not None:
                new_axes[n] = chunks[ax]
            else:
                new_axes[n] = 1
        out_ind = tuple(temp_out_ind)
        if max(new_axis) > max(out_ind):
            raise ValueError("New_axis values do not fill in all dimensions")

    if chunks is not None:
        if len(chunks) != len(out_ind):
            raise ValueError(
                f"Provided chunks have {len(chunks)} dims; expected {len(out_ind)} dims"
            )
        adjust_chunks = dict(zip(out_ind, chunks))
    else:
        adjust_chunks = None

    return blockwise(
        func,
        out_ind,
        *concat(argpairs),
        dtype=dtype,
        adjust_chunks=adjust_chunks,
        new_axes=new_axes,
        align_arrays=False,
        **kwargs,
    )


def map_direct(
    func, *args: "Array", shape, dtype, chunks, extra_projected_mem, spec=None, **kwargs
) -> "Array":
    """
    Apply a function across blocks of a new array, reading directly from side inputs (not necessarily in a blockwise fashion).

    Parameters
    ----------
    func : callable
        Function to apply to every block to produce the output array.
        Must accept ``block_id`` as a keyword argument (with same meaning as for ``map_blocks``).
    args : arrays
        The side-input arrays that may be accessed directly in the function.
    shape : tuple
        Shape of the output array.
    dtype : np.dtype
        The ``dtype`` of the output array.
    chunks : tuple
        Chunk shape of blocks in the output array.
    extra_projected_mem : int
        Extra memory projected to be needed (in bytes) for each map task. This should take into account the
        memory allocations for any reads from the side-input arrays (``args``).
    spec : Spec
        Specification for the new array. If not specified, the one from the first side input
        (`args`) will be used (if any).
    """
    warn(
        "`map_direct` is pending deprecation, please use `map_selection` instead",
        PendingDeprecationWarning,
        stacklevel=2,
    )

    from cubed.array_api.creation_functions import empty_virtual_array

    if spec is None and len(args) > 0 and hasattr(args[0], "spec"):
        spec = args[0].spec

    out = empty_virtual_array(shape, dtype=dtype, chunks=chunks, spec=spec)

    kwargs["arrays"] = args

    def new_func(func):
        def wrap(*a, block_id=None, **kw):
            arrays = kw.pop("arrays")
            args = a + arrays
            return func(*args, block_id=block_id, **kw)

        return wrap

    return map_blocks(
        new_func(func),
        out,
        dtype=dtype,
        chunks=chunks,
        extra_source_arrays=args,
        extra_projected_mem=extra_projected_mem,
        fusable_with_predecessors=False,  # don't allow fusion with predecessors since side inputs are not accounted for
        **kwargs,
    )


def rechunk(x, chunks, *, target_store=None, min_mem=None, use_new_impl=False):
    """Change the chunking of an array without changing its shape or data.

    Parameters
    ----------
    chunks : tuple
        The desired chunks of the array after rechunking.

    Returns
    -------
    cubed.Array
        An array with the desired chunks.
    """
    if use_new_impl:
        return rechunk_new(x, chunks, min_mem=min_mem)

    if isinstance(chunks, dict):
        chunks = {validate_axis(c, x.ndim): v for c, v in chunks.items()}
        for i in range(x.ndim):
            if i not in chunks:
                chunks[i] = x.chunks[i]
            elif chunks[i] is None:
                chunks[i] = x.chunks[i]
    if isinstance(chunks, (tuple, list)):
        chunks = tuple(lc if lc is not None else rc for lc, rc in zip(chunks, x.chunks))

    normalized_chunks = normalize_chunks(chunks, x.shape, dtype=x.dtype)
    if x.chunks == normalized_chunks:
        return x
    # normalizing takes care of dict args for chunks
    target_chunks = to_chunksize(normalized_chunks)

    # merge chunks special case
    if all(c1 % c0 == 0 for c0, c1 in zip(x.chunksize, target_chunks)):
        return merge_chunks(x, target_chunks)

    name = gensym()
    spec = x.spec
    if target_store is None:
        target_store = new_temp_path(name=name, spec=spec)
    name_int = f"{name}-int"
    temp_store = new_temp_path(name=name_int, spec=spec)
    ops = primitive_rechunk(
        x.zarray_maybe_lazy,
        source_array_name=name,
        int_array_name=name_int,
        target_chunks=target_chunks,
        allowed_mem=spec.allowed_mem,
        reserved_mem=spec.reserved_mem,
        target_store=target_store,
        temp_store=temp_store,
        storage_options=spec.storage_options,
    )

    from cubed.array_api import Array

    if len(ops) == 1:
        op = ops[0]
        plan = Plan._new(
            name,
            "rechunk",
            op.target_array,
            op,
            False,
            x,
        )
        return Array(name, op.target_array, spec, plan)

    else:
        op1 = ops[0]
        plan1 = Plan._new(
            name_int,
            "rechunk",
            op1.target_array,
            op1,
            False,
            x,
        )
        x_int = Array(name_int, op1.target_array, spec, plan1)

        op2 = ops[1]
        plan2 = Plan._new(
            name,
            "rechunk",
            op2.target_array,
            op2,
            False,
            x_int,
        )
        return Array(name, op2.target_array, spec, plan2)


def rechunk_new(x, chunks, *, min_mem=None):
    """Change the chunking of an array without changing its shape or data.

    Parameters
    ----------
    chunks : tuple
        The desired chunks of the array after rechunking.

    Returns
    -------
    cubed.Array
        An array with the desired chunks.
    """
    out = x
    for copy_chunks, target_chunks in _rechunk_plan(x, chunks, min_mem=min_mem):
        out = _rechunk(out, copy_chunks, target_chunks)
    return out


def _rechunk_plan(x, chunks, *, min_mem=None):
    if isinstance(chunks, dict):
        chunks = {validate_axis(c, x.ndim): v for c, v in chunks.items()}
        for i in range(x.ndim):
            if i not in chunks:
                chunks[i] = x.chunks[i]
            elif chunks[i] is None:
                chunks[i] = x.chunks[i]
    if isinstance(chunks, (tuple, list)):
        chunks = tuple(lc if lc is not None else rc for lc, rc in zip(chunks, x.chunks))

    normalized_chunks = normalize_chunks(chunks, x.shape, dtype=x.dtype)
    if x.chunks == normalized_chunks:
        return x
    # normalizing takes care of dict args for chunks
    target_chunks = to_chunksize(normalized_chunks)

    # merge chunks special case
    if all(c1 % c0 == 0 for c0, c1 in zip(x.chunksize, target_chunks)):
        return merge_chunks(x, target_chunks)

    spec = x.spec
    source_chunks = to_chunksize(normalize_chunks(x.chunks, x.shape, dtype=x.dtype))

    # rechunker doesn't take account of uncompressed and compressed copies of the
    # input and output array chunk/selection, so adjust appropriately:
    #  1 input array plus copies to read that array from storage,
    #  1 array for processing,
    #  1 output array plus copies to write that array to storage
    buffer_copies = get_buffer_copies(spec)
    total_copies = 1 + buffer_copies.read + 1 + 1 + buffer_copies.write
    rechunker_max_mem = (spec.allowed_mem - spec.reserved_mem) // total_copies
    if min_mem is None:
        min_mem = min(rechunker_max_mem // 20, x.nbytes)
    stages = multistage_rechunking_plan(
        shape=x.shape,
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        itemsize=x.dtype.itemsize,
        min_mem=min_mem,
        max_mem=rechunker_max_mem,
    )

    for i, stage in enumerate(stages):
        last_stage = i == len(stages) - 1
        read_chunks, int_chunks, write_chunks = stage

        # Use target chunks for last stage
        target_chunks_ = target_chunks if last_stage else write_chunks

        if read_chunks == write_chunks:
            yield read_chunks, target_chunks_
        else:
            yield read_chunks, int_chunks
            yield write_chunks, target_chunks_


def _rechunk(x, copy_chunks, target_chunks):
    # rechunk x so that its target store has target_chunks, using copy_chunks as the size of chunks for copying from source to target

    normalized_copy_chunks = normalize_chunks(copy_chunks, x.shape, dtype=x.dtype)
    copy_chunks = to_chunksize(normalized_copy_chunks)

    copy_chunks_mem = array_memory(x.dtype, copy_chunks)

    target_chunks = normalize_chunks(target_chunks, x.shape, dtype=x.dtype)
    target_chunks = to_chunksize(target_chunks)

    def selection_function(out_key):
        out_coords = out_key[1:]
        return get_item(normalized_copy_chunks, out_coords)

    max_num_input_blocks = math.prod(
        math.ceil(c1 / c0) for c0, c1 in zip(x.chunksize, copy_chunks)
    )

    return map_selection(
        None,  # no function to apply after selection
        selection_function,
        x,
        x.shape,
        x.dtype,
        normalized_copy_chunks,
        max_num_input_blocks=max_num_input_blocks,
        target_chunks_=target_chunks,
        fusable_with_predecessors=False,
        fusable_with_successors=False,
        op_name="rechunk",
        extra_projected_mem=copy_chunks_mem,
    )


def merge_chunks(x, chunks):
    """Merge multiple chunks into one."""
    target_chunksize = chunks
    if len(target_chunksize) != x.ndim:
        raise ValueError(
            f"Chunks {target_chunksize} must have same number of dimensions as array ({x.ndim})"
        )
    if not all(c1 % c0 == 0 for c0, c1 in zip(x.chunksize, target_chunksize)):
        raise ValueError(
            f"Chunks {target_chunksize} must be a multiple of array's chunks {x.chunksize}"
        )

    target_chunks = normalize_chunks(chunks, x.shape, dtype=x.dtype)

    def selection_function(out_key):
        out_coords = out_key[1:]
        return get_item(target_chunks, out_coords)

    max_num_input_blocks = math.prod(
        c1 // c0 for c0, c1 in zip(x.chunksize, target_chunksize)
    )

    return map_selection(
        None,  # no function to apply after selection
        selection_function,
        x,
        x.shape,
        x.dtype,
        target_chunks,
        max_num_input_blocks=max_num_input_blocks,
    )


def reduction(
    x: "Array",
    func,
    combine_func=None,
    aggregate_func=None,
    axis=None,
    intermediate_dtype=None,
    dtype=None,
    keepdims=False,
    split_every=None,
    combine_sizes=None,
    extra_func_kwargs=None,
) -> "Array":
    """Apply a function to reduce an array along one or more axes.

    Parameters
    ----------
    x: Array
        Array being reduced along one or more axes.
    func: callable
        Function to apply to each chunk of data before reduction.
    combine_func: callable, optional
        Function which may be applied recursively to intermediate chunks of
        data. The number of chunks that are combined in each round is
        determined by the ``split_every`` parameter. The output of the
        function is a chunk with size one (or the size specified in
        ``combine_sizes``) in each of the reduction axes. If omitted,
        it defaults to ``func``.
    aggregate_func: callable, optional
        Function to apply to each of the final chunks to produce the final output.
    axis: int or sequence of ints, optional
        Axis or axes to aggregate upon. If omitted, aggregate along all axes.
    intermediate_dtype: dtype
        Data type of intermediate output.
    dtype: dtype
        Data type of output.
    keepdims: boolean, optional
        Whether the reduction function should preserve the reduced axes,
        or remove them.
    split_every: int >= 2 or dict(axis: int), optional
        The number of chunks to combine in one round along each axis in the
        recursive aggregation.
    combine_sizes: dict(axis: int), optional
        The resulting size of each axis after reduction. Each reduction axis
        defaults to size one if not specified.
    extra_func_kwargs: dict, optional
        Extra keyword arguments to pass to ``func`` and ``combine_func``.
    """
    if combine_func is None:
        if func is None:
            raise ValueError(
                "At least one of `func` and `combine_func` must be specified in reduction"
            )
        combine_func = func
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)
    if intermediate_dtype is None:
        intermediate_dtype = dtype

    split_every = _normalize_split_every(split_every, axis)

    if func is None:
        initial_func = None
    else:
        initial_func = partial(
            func, axis=axis, keepdims=True, **(extra_func_kwargs or {})
        )
    result = partial_reduce(
        x,
        partial(combine_func, **(extra_func_kwargs or {})),
        initial_func=initial_func,
        split_every=split_every,
        dtype=intermediate_dtype,
        combine_sizes=combine_sizes,
    )

    # combine intermediates
    result = tree_reduce(
        result,
        partial(combine_func, **(extra_func_kwargs or {})),
        axis=axis,
        dtype=intermediate_dtype,
        split_every=split_every,
        combine_sizes=combine_sizes,
    )

    # aggregate final chunks
    if aggregate_func is not None:
        result = map_blocks(
            partial(aggregate_func, **(extra_func_kwargs or {})), result, dtype=dtype
        )

    if not keepdims:
        axis_to_squeeze = tuple(i for i in axis if result.shape[i] == 1)
        if len(axis_to_squeeze) > 0:
            result = squeeze(result, axis_to_squeeze)

    from cubed.array_api import astype

    result = astype(result, dtype, copy=False)

    return result


def _normalize_split_every(split_every, axis):
    split_every = split_every or 4
    if isinstance(split_every, dict):
        split_every = {k: split_every.get(k, 2) for k in axis}
    elif isinstance(split_every, Integral):
        n = builtins.max(int(split_every ** (1 / (len(axis) or 1))), 2)
        split_every = dict.fromkeys(axis, n)
    else:
        raise ValueError("split_every must be a int or a dict")
    return split_every


def tree_reduce(
    x,
    func,
    axis,
    dtype,
    split_every=None,
    combine_sizes=None,
):
    """Apply a reduction function repeatedly across multiple axes."""
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)

    split_every = _normalize_split_every(split_every, axis)

    depth = 0
    for i, n in enumerate(x.numblocks):
        if i in split_every and split_every[i] != 1:
            depth = int(builtins.max(depth, math.ceil(math.log(n, split_every[i]))))
    for _ in range(depth):
        x = partial_reduce(
            x,
            func,
            split_every=split_every,
            dtype=dtype,
            combine_sizes=combine_sizes,
        )
    return x


def partial_reduce(
    x,
    func,
    initial_func=None,
    split_every=None,
    dtype=None,
    combine_sizes=None,
):
    """Apply a reduction function to multiple blocks across multiple axes.

    Parameters
    ----------
    x: Array
        Array being reduced along one or more axes
    func: callable
        Reduction function to apply to each chunk of data, resulting in a chunk
        with size one (or the size specified in ``combine_sizes``) in each of
        the reduction axes.
    initial_func: callable, optional
        Function to apply to each chunk of data before reduction.
    split_every: int >= 2 or dict(axis: int), optional
        The number of chunks to combine in one round along each axis in the
        recursive aggregation.
    dtype: dtype
        Output data type.
    combine_sizes: dict(axis: int), optional
        The resulting size of each axis after reduction. Each reduction axis
        defaults to size one if not specified.
    """
    # map over output chunks
    axis = tuple(ax for ax in split_every.keys())
    combine_sizes = combine_sizes or {}
    combine_sizes = {k: combine_sizes.get(k, 1) for k in axis}
    chunks = tuple(
        (combine_sizes[i],) * math.ceil(len(c) / split_every[i])
        if i in split_every
        else c
        for (i, c) in enumerate(x.chunks)
    )
    shape = tuple(map(sum, chunks))

    def key_function(out_key):
        out_coords = out_key[1:]

        # return a tuple with a single item that is an iterator of input keys to be merged
        in_keys = [
            list(
                range(
                    bi * split_every.get(i, 1),
                    min((bi + 1) * split_every.get(i, 1), x.numblocks[i]),
                )
            )
            for i, bi in enumerate(out_coords)
        ]
        return (iter([(x.name,) + tuple(p) for p in product(*in_keys)]),)

    # Since key_function returns an iterator of input keys, the the array chunks passed to
    # _partial_reduce are retrieved one at a time. However, we need an extra chunk of memory
    # to stay within limits (maybe because the iterator doesn't free the previous object
    # before getting the next). We also need extra memory to hold two reduced chunks, since
    # they are concatenated two at a time.
    extra_projected_mem = x.chunkmem + 2 * array_memory(dtype, to_chunksize(chunks))

    return general_blockwise(
        _partial_reduce,
        key_function,
        x,
        shapes=[shape],
        dtypes=[dtype],
        chunkss=[chunks],
        extra_projected_mem=extra_projected_mem,
        num_input_blocks=(sum(split_every.values()),),
        iterable_input_blocks=(True,),
        reduce_func=func,
        initial_func=initial_func,
        axis=axis,
    )


def _partial_reduce(arrays, reduce_func=None, initial_func=None, axis=None):
    # reduce each array in turn, accumulating in result
    assert not isinstance(
        arrays, list
    ), "partial reduce expects an iterator of array blocks, not a list"
    result = None
    for array in arrays:
        if initial_func is not None:
            array = initial_func(array)
        reduced_chunk = reduce_func(array, axis=axis, keepdims=True)
        if result is None:
            result = reduced_chunk
        elif isinstance(result, dict):
            assert result.keys() == reduced_chunk.keys()
            result = {
                # only need to concatenate along first axis
                k: nxp.concat([result[k], reduced_chunk[k]], axis=axis[0])
                for k in result.keys()
            }
            result = reduce_func(result, axis=axis, keepdims=True)
        else:
            # only need to concatenate along first axis
            result = nxp.concat([result, reduced_chunk], axis=axis[0])
            result = reduce_func(result, axis=axis, keepdims=True)

    return result


def arg_reduction(x, /, arg_func, axis=None, *, keepdims=False, split_every=None):
    """A reduction that returns the array indexes, not the values."""
    dtype = nxp.int64  # index data type
    intermediate_dtype = [("i", dtype), ("v", x.dtype)]

    # initial map does arg reduction on each block, and uses block id to find the absolute index within whole array
    chunks = tuple((1,) * len(c) if i == axis else c for i, c in enumerate(x.chunks))
    out = map_blocks(
        partial(_arg_map_func, arg_func=arg_func),
        x,
        dtype=intermediate_dtype,
        chunks=chunks,
        axis=axis,
        size=to_chunksize(x.chunks)[axis],
    )

    # then reduce across blocks
    return reduction(
        out,
        _arg_func,
        combine_func=partial(_arg_combine, arg_func=arg_func),
        aggregate_func=_arg_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
    )


def _arg_map_func(a, axis, arg_func=None, size=None, block_id=None):
    i = arg_func(a, axis=axis, keepdims=True)
    # note that the array API doesn't have take_along_axis, so this may fail
    v = nxp.take_along_axis(a, i, axis=axis)
    # add block offset to i so it is absolute index within whole array
    offset = block_id[axis] * size
    return {"i": i + offset, "v": v}


def _arg_func(a, **kwargs):
    # pass through
    return {"i": a["i"], "v": a["v"]}


def _arg_combine(a, arg_func=None, **kwargs):
    # convert axis from single value tuple to int
    axis = kwargs.pop("axis")[0]

    i = a["i"]
    v = a["v"]

    # find indexes of values in v and apply to i and v
    vi = arg_func(v, axis=axis, **kwargs)
    # note that the array API doesn't have take_along_axis, so this may fail
    i_combined = nxp.take_along_axis(i, vi, axis=axis)
    v_combined = nxp.take_along_axis(v, vi, axis=axis)
    return {"i": i_combined, "v": v_combined}


def _arg_aggregate(a):
    # just return index values
    return a["i"]


def squeeze(x, /, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)

    if any(x.shape[i] != 1 for i in axis):
        raise ValueError("cannot squeeze axis with size other than one")

    axis = validate_axis(axis, x.ndim)

    chunks = tuple(c for i, c in enumerate(x.chunks) if i not in axis)

    return map_blocks(
        nxp.squeeze, x, dtype=x.dtype, chunks=chunks, drop_axis=axis, axis=axis
    )


def unify_chunks(*args: "Array", **kwargs):
    # From dask unify_chunks
    if not args:
        return {}, []

    arginds = [(a, ind) for a, ind in partition(2, args)]  # [x, ij, y, jk]

    arrays, inds = zip(*arginds)
    if all(ind is None for ind in inds):
        return {}, list(arrays)
    if all(ind == inds[0] for ind in inds) and all(
        a.chunks == arrays[0].chunks for a in arrays
    ):
        return dict(zip(inds[0], arrays[0].chunks)), arrays

    nameinds = []
    blockdim_dict = dict()
    max_parts = 0
    for a, ind in arginds:
        if ind is not None:
            nameinds.append((a.name, ind))
            blockdim_dict[a.name] = a.chunks
            max_parts = max(max_parts, a.npartitions)
        else:
            nameinds.append((a, ind))

    chunkss = broadcast_dimensions(
        nameinds, blockdim_dict, consolidate=smallest_blockdim
    )

    arrays = []
    for a, i in arginds:
        if i is None:
            arrays.append(a)
        else:
            chunks = tuple(
                chunkss[j]
                if a.shape[n] > 1
                else (a.shape[n],)
                if not np.isnan(sum(chunkss[j]))
                else None
                for n, j in enumerate(i)
            )
            if chunks != a.chunks and all(a.chunks):
                # this will raise if chunks are not regular
                # but this should never happen with smallest_blockdim
                chunksize = to_chunksize(chunks)  # type: ignore
                arrays.append(rechunk(a, chunksize))
            else:
                arrays.append(a)
    return chunkss, arrays


def smallest_blockdim(blockdims):
    """Find the smallest block dimensions from the list of block dimensions

    Unlike Dask's common_blockdim, this returns regular chunks (assuming
    regular chunks are passed in).
    """
    if not any(blockdims):
        return ()
    non_trivial_dims = {d for d in blockdims if len(d) > 1}
    if len(non_trivial_dims) == 1:
        return first(non_trivial_dims)
    if len(non_trivial_dims) == 0:
        return max(blockdims, key=first)

    if len(set(map(sum, non_trivial_dims))) > 1:
        raise ValueError("Chunks do not add up to same value", blockdims)

    # find dims with the smallest first chunk
    m = -1
    out = None
    for ntd in non_trivial_dims:
        if m == -1 or ntd[0] < m:
            m = ntd[0]
            out = ntd
    return out
