import numbers
from functools import partial
from numbers import Integral, Number
from operator import add

import numpy as np
import zarr
from dask.array.core import common_blockdim, normalize_chunks
from dask.array.utils import validate_axis
from dask.blockwise import broadcast_dimensions
from dask.utils import has_keyword
from tlz import concat, partition
from toolz import accumulate, map
from zarr.indexing import BasicIndexer, IntDimIndexer, is_slice, replace_ellipsis

from cubed.core.array import CoreArray, check_array_specs, compute, gensym
from cubed.core.plan import Plan, new_temp_store
from cubed.primitive.blockwise import blockwise as primitive_blockwise
from cubed.primitive.rechunk import rechunk as primitive_rechunk
from cubed.utils import chunk_memory, get_item, to_chunksize


def from_array(x, chunks="auto", asarray=None, spec=None):
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
        name = gensym()
        target = x
        plan = Plan(name, "from_array", target)
        arr = CoreArray._new(name, target, spec, plan)

        chunksize = to_chunksize(outchunks)
        if chunks != "auto" and previous_chunks != chunksize:
            arr = rechunk(arr, chunksize)
        return arr

    if asarray is None:
        asarray = not hasattr(x, "__array_function__")

    return map_direct(
        _from_array,
        x,
        shape=x.shape,
        dtype=x.dtype,
        chunks=outchunks,
        extra_required_mem=0,
        spec=spec,
        outchunks=outchunks,
        asarray=asarray,
    )


def _from_array(e, x, outchunks=None, asarray=None, block_id=None):
    out = x[get_item(outchunks, block_id)]
    if asarray:
        out = np.asarray(out)
    return out


def from_zarr(store, spec=None):
    """Load an array from Zarr storage.

    Parameters
    ----------
    store : string
        Path to input Zarr store
    spec : cubed.Spec, optional
        The spec to use for the computation.

    Returns
    -------
    cubed.CoreArray
        The array loaded from Zarr storage.
    """
    name = gensym()
    target = zarr.open(store, mode="r")

    plan = Plan(name, "from_zarr", target)
    return CoreArray._new(name, target, spec, plan)


def store(sources, targets, executor=None, **kwargs):
    """Save source arrays to array-like objects.

    In the current implementation ``targets`` must be Zarr arrays.

    Note that this operation is eager, and will run the computation
    immediately.

    Parameters
    ----------
    x : cubed.CoreArray or collection of cubed.CoreArray
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


def to_zarr(x, store, executor=None, **kwargs):
    """Save an array to Zarr storage.

    Note that this operation is eager, and will run the computation
    immediately.

    Parameters
    ----------
    x : cubed.CoreArray
        Array to save
    store : string
        Path to output Zarr store
    executor : cubed.runtime.types.Executor, optional
        The executor to use to run the computation.
        Defaults to using the in-process Python executor.
    """
    # Note that the intermediate write to x's store will be optimized away
    # by map fusion (if it was produced with a blockwise operation).
    identity = lambda a: a
    ind = tuple(range(x.ndim))
    out = blockwise(
        identity, ind, x, ind, dtype=x.dtype, align_arrays=False, target_store=store
    )
    out.compute(executor=executor, _return_in_memory_array=False, **kwargs)


def blockwise(
    func,
    out_ind,
    *args,
    dtype=None,
    adjust_chunks=None,
    new_axes=None,
    align_arrays=True,
    target_store=None,
    **kwargs,
):
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
    chunks = tuple(chunks)
    shape = tuple(map(sum, chunks))

    # replace arrays with zarr arrays
    zargs = list(args)
    zargs[::2] = [a.zarray for a in arrays]

    extra_source_arrays = kwargs.pop("extra_source_arrays", [])
    source_arrays = list(arrays) + list(extra_source_arrays)

    extra_required_mem = kwargs.pop("extra_required_mem", 0)

    name = gensym()
    spec = check_array_specs(arrays)
    if target_store is None:
        target_store = new_temp_store(name=name, spec=spec)
    pipeline, target, required_mem, num_tasks = primitive_blockwise(
        func,
        out_ind,
        *zargs,
        max_mem=spec.max_mem,
        target_store=target_store,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        new_axes=new_axes,
        **kwargs,
    )
    plan = Plan(
        name,
        "blockwise",
        target,
        pipeline,
        required_mem + extra_required_mem,
        num_tasks,
        *source_arrays,
    )
    return CoreArray._new(name, target, spec, plan)


def elemwise(func, *args, dtype=None):
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
    if not isinstance(key, tuple):
        key = (key,)

    # No op case
    if all(is_slice(ind) and ind == slice(None) for ind in key):
        return x

    # Remove None values, to be filled in with expand_dims at end
    where_none = [i for i, ind in enumerate(key) if ind is None]
    for i, a in enumerate(where_none):
        n = sum(isinstance(ind, Integral) for ind in key[:a])
        if n:
            where_none[i] -= n
    key = tuple(ind for ind in key if ind is not None)

    # Replace ellipsis with slices
    selection = replace_ellipsis(key, x.shape)

    # Use a Zarr BasicIndexer just to find the resulting array shape and chunks
    indexer = BasicIndexer(selection, x.zarray)

    shape = indexer.shape
    dtype = x.dtype
    chunks = tuple(
        s.dim_chunk_len
        for s in indexer.dim_indexers
        if not isinstance(s, IntDimIndexer)
    )
    chunks = normalize_chunks(chunks, shape, dtype=dtype)

    assert all(isinstance(s, (slice, Integral)) for s in selection)
    if any(s.step is not None and s.step != 1 for s in selection if is_slice(s)):
        raise NotImplementedError(f"Slice step must be 1: {key}")

    # memory allocated by reading one chunk from input array
    # note that although the output chunk will overlap multiple input chunks, zarr will
    # read the chunks in series, reusing the buffer
    extra_required_mem = x.chunkmem

    out = map_direct(
        _read_index_chunk,
        x,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_required_mem=extra_required_mem,
        target_chunks=chunks,
        selection=selection,
    )

    for axis in where_none:
        from cubed.array_api.manipulation_functions import expand_dims

        out = expand_dims(out, axis=axis)

    return out


def _read_index_chunk(x, *arrays, target_chunks=None, selection=None, block_id=None):
    array = arrays[0]
    idx = block_id
    out = array.zarray[_integers_and_slices_selection(target_chunks, idx, selection)]
    return out


def _integers_and_slices_selection(target_chunks, idx, selection):
    # integer and slice indexes can be interspersed in selection
    # idx is the chunk index for the output (target_chunks)

    # these are just for slices
    offsets = tuple(s.start or 0 for s in selection if is_slice(s))
    starts = tuple(tuple(accumulate(add, c, o)) for c, o in zip(target_chunks, offsets))

    sel = []
    slice_count = 0  # keep track of slices in selection
    for s in selection:
        if is_slice(s):
            i = idx[slice_count]
            start = starts[slice_count]
            sel.append(slice(start[i], start[i + 1]))
            slice_count += 1
        else:
            sel.append(s)
    return tuple(sel)


def map_blocks(
    func, *args, dtype=None, chunks=None, drop_axis=[], new_axis=None, **kwargs
):
    """Apply a function to corresponding blocks from multiple input arrays."""
    if has_keyword(func, "block_id"):
        from cubed.array_api import asarray

        # Create an array of index offsets with the same chunk structure as the args,
        # which we convert to block ids (chunk coordinates) later.
        a = args[0]
        offsets = np.arange(a.npartitions, dtype=np.int32).reshape(a.numblocks)
        offsets = asarray(offsets, spec=a.spec)
        # rechunk in a separate operation to avoid potentially writing lots of zarr chunks from the client
        offsets_chunks = (1,) * len(a.numblocks)
        offsets = rechunk(offsets, offsets_chunks)
        new_args = args + (offsets,)

        def offset_to_block_id(offset):
            return list(np.ndindex(*(a.numblocks)))[offset]

        def func_with_block_id(func):
            def wrap(*a, **kw):
                block_id = offset_to_block_id(a[-1].item())
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
    func, *args, dtype=None, chunks=None, drop_axis=[], new_axis=None, **kwargs
):
    # based on dask

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
        out_ind = list(out_ind)
        for ax in sorted(new_axis):
            n = len(out_ind) + len(drop_axis)
            out_ind.insert(ax, n)
            if chunks is not None:
                new_axes[n] = chunks[ax]
            else:
                new_axes[n] = 1
        out_ind = tuple(out_ind)
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
    func, *args, shape, dtype, chunks, extra_required_mem, spec=None, **kwargs
):
    """
    Map a function across blocks of a new array, using side-input arrays to read directly from.

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
    extra_required_mem : int
        Extra memory required (in bytes) for each map task. This should take into account the
        memory requirements for any reads from the side-input arrays (``args``).
    spec : Spec
        Specification for the new array. If not specified, the one from the first side input
        (`args`) will be used (if any).
    """

    from cubed.array_api.creation_functions import empty

    if spec is None and len(args) > 0 and hasattr(args[0], "spec"):
        spec = args[0].spec

    out = empty(shape, dtype=dtype, chunks=chunks, spec=spec)

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
        extra_required_mem=extra_required_mem,
        **kwargs,
    )


def rechunk(x, chunks, target_store=None):
    name = gensym()
    spec = x.spec
    if target_store is None:
        target_store = new_temp_store(name=name, spec=spec)
    temp_store = new_temp_store(name=f"{name}-intermediate", spec=spec)
    pipeline, target, required_mem, num_tasks = primitive_rechunk(
        x.zarray,
        target_chunks=chunks,
        max_mem=spec.max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )
    plan = Plan(name, "rechunk", target, pipeline, required_mem, num_tasks, x)
    return CoreArray._new(name, target, spec, plan)


def reduction(
    x,
    func,
    combine_func=None,
    aggegrate_func=None,
    axis=None,
    intermediate_dtype=None,
    dtype=None,
    keepdims=False,
):
    if combine_func is None:
        combine_func = func
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)
    if intermediate_dtype is None:
        intermediate_dtype = dtype

    inds = tuple(range(x.ndim))

    result = x
    max_mem = x.spec.max_mem

    # reduce initial chunks (if any axis chunksize is > 1)
    if (
        any(s > 1 for i, s in enumerate(result.chunksize) if i in axis)
        or func != combine_func
    ):
        args = (result, inds)
        adjust_chunks = {
            i: (1,) * len(c) if i in axis else c for i, c in enumerate(result.chunks)
        }
        result = blockwise(
            func,
            inds,
            *args,
            axis=axis,
            keepdims=True,
            dtype=intermediate_dtype,
            adjust_chunks=adjust_chunks,
        )

    # rechunk/reduce along axis in multiple rounds until there's a single block in each reduction axis
    while any(n > 1 for i, n in enumerate(result.numblocks) if i in axis):
        # rechunk along axis
        target_chunks = list(result.chunksize)
        chunk_mem = chunk_memory(intermediate_dtype, result.chunksize)
        for i, s in enumerate(result.shape):
            if i in axis:
                if len(axis) > 1:
                    # TODO: make sure chunk size doesn't exceed max_mem for multi-axis reduction
                    target_chunks[i] = s
                else:
                    # factor of 4 is memory for {compressed, uncompressed} x {input, output} (see rechunk.py)
                    target_chunk_size = (max_mem - chunk_mem) // (chunk_mem * 4)
                    if target_chunk_size <= 1:
                        raise ValueError(
                            f"Not enough memory for reduction. Increase max_mem ({max_mem}) or decrease chunk size"
                        )
                    target_chunks[i] = min(s, target_chunk_size)
        target_chunks = tuple(target_chunks)
        result = rechunk(result, target_chunks)

        # reduce chunks (if any axis chunksize is > 1)
        if any(s > 1 for i, s in enumerate(result.chunksize) if i in axis):
            args = (result, inds)
            adjust_chunks = {
                i: (1,) * len(c) if i in axis else c
                for i, c in enumerate(result.chunks)
            }
            result = blockwise(
                combine_func,
                inds,
                *args,
                axis=axis,
                keepdims=True,
                dtype=intermediate_dtype,
                adjust_chunks=adjust_chunks,
            )

    if aggegrate_func is not None:
        result = map_blocks(aggegrate_func, result, dtype=dtype)

    if not keepdims:
        axis_to_squeeze = tuple(i for i in axis if result.shape[i] == 1)
        if len(axis_to_squeeze) > 0:
            result = squeeze(result, axis_to_squeeze)

    from cubed.array_api import astype

    result = astype(result, dtype, copy=False)

    return result


def arg_reduction(x, /, arg_func, axis=None, *, keepdims=False):
    dtype = np.int64  # index data type
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
        aggegrate_func=_arg_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
    )


def _arg_map_func(a, axis, arg_func=None, size=None, block_id=None):
    i = arg_func(a, axis=axis, keepdims=True)
    v = np.take_along_axis(a, i, axis=axis)
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
    i_combined = np.take_along_axis(i, vi, axis=axis)
    v_combined = np.take_along_axis(v, vi, axis=axis)
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
        np.squeeze, x, dtype=x.dtype, chunks=chunks, drop_axis=axis, axis=axis
    )


def unify_chunks(*args, **kwargs):
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

    chunkss = broadcast_dimensions(nameinds, blockdim_dict, consolidate=common_blockdim)

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
                chunksize = to_chunksize(chunks)
                arrays.append(rechunk(a, chunksize))
            else:
                arrays.append(a)
    return chunkss, arrays
