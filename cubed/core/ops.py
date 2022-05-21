import numbers
from math import prod
from numbers import Integral, Number

import numpy as np
import zarr
from dask.array.core import common_blockdim, normalize_chunks
from dask.array.utils import validate_axis
from dask.blockwise import broadcast_dimensions
from tlz import concat, partition
from toolz import map

from cubed.core.array import Array, Plan, gensym, new_temp_store
from cubed.primitive import blockwise as primitive_blockwise
from cubed.primitive import rechunk as primitive_rechunk
from cubed.utils import to_chunksize


def from_zarr(store, spec=None):
    """Load an array from Zarr storage."""
    name = gensym()
    target = zarr.open(store, mode="r")

    plan = Plan(name, "from_zarr", target, spec)
    return Array(name, target, plan)


def to_zarr(x, store, return_stored=False, executor=None):
    """Save an array to Zarr storage."""
    # Use rechunk with same chunks to implement a straight copy.
    # It would be good to avoid this copy in the future. Maybe allow optional store to be passed to all functions?
    # Zarr views still need to be copied to materialize them, however.
    out = rechunk(x, x.chunksize, target_store=store)
    return out.compute(return_stored=return_stored, executor=executor)


def blockwise(
    func, out_ind, *args, dtype=None, adjust_chunks=None, align_arrays=True, **kwargs
):
    arrays = args[::2]

    assert len(arrays) > 0

    # Calculate output chunking. A lot of this is from dask's blockwise function
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

    name = gensym()
    spec = arrays[0].plan.spec
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
        **kwargs,
    )
    plan = Plan(
        name, "blockwise", target, spec, pipeline, required_mem, num_tasks, *arrays
    )
    return Array(name, target, plan)


def elementwise_unary_operation(x, func, dtype):
    return map_blocks(func, x, dtype=dtype)


def elementwise_binary_operation(x1, x2, func, dtype):
    # TODO: check x1 and x2 are compatible

    return map_blocks(func, x1, x2, dtype=dtype)


def map_blocks(func, *args, dtype=None, chunks=None, drop_axis=[], **kwargs):
    # based on dask
    if isinstance(drop_axis, Number):
        drop_axis = [drop_axis]

    arrs = args
    argpairs = [
        (a, tuple(range(a.ndim))[::-1]) if isinstance(a, Array) else (a, None)
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
        **kwargs,
    )


def rechunk(x, chunks, target_store=None):
    name = gensym()
    spec = x.plan.spec
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
    plan = Plan(name, "rechunk", target, spec, pipeline, required_mem, num_tasks, x)
    return Array(name, target, plan)


def reduction(x, func, combine_func=None, axis=None, dtype=None, keepdims=False):
    if combine_func is None:
        combine_func = func
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, Integral):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)

    inds = tuple(range(x.ndim))

    result = x
    max_mem = x.plan.spec.max_mem

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
            dtype=dtype,
            adjust_chunks=adjust_chunks,
        )

    # rechunk/reduce along axis in multiple rounds until there's a single block in each reduction axis
    while any(n > 1 for i, n in enumerate(result.numblocks) if i in axis):
        # rechunk along axis
        target_chunks = list(result.chunksize)
        chunk_mem = np.dtype(dtype).itemsize * prod(result.chunksize)
        for i, s in enumerate(result.shape):
            if i in axis:
                if len(axis) > 1:
                    # TODO: make sure chunk size doesn't exceed max_mem for multi-axis reduction
                    target_chunks[i] = s
                else:
                    target_chunks[i] = min(s, (max_mem - chunk_mem) // chunk_mem)
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
                dtype=dtype,
                adjust_chunks=adjust_chunks,
            )

    # TODO: [optimization] remove extra squeeze (and materialized zarr) by doing it as a part of the last blockwise
    if not keepdims:
        result = squeeze(result, axis)

    return result


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
