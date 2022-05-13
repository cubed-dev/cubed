from typing import Iterable

import numpy as np
import zarr
from dask.array.core import normalize_chunks
from dask.array.reductions import numel

from barry.primitive import broadcast_to as primitive_broadcast_to
from barry.utils import to_chunksize

from .core import (
    Array,
    Plan,
    blockwise,
    elementwise_binary_operation,
    elementwise_unary_operation,
    gensym,
    map_blocks,
    new_temp_store,
    new_temp_zarr,
    reduction,
    squeeze,
)

# Creation functions


def _arange(a, size):
    start = a[0]
    return np.arange(start * size, (start + 1) * size)


def arange(
    start, /, stop=None, step=1, *, dtype=None, device=None, chunks="auto", spec=None
):
    # TODO: implement step
    # TODO: support array length that isn't a multiple of chunks
    if stop is None:
        start, stop = 0, start
    num = int(max(np.ceil((stop - start) / step), 0))
    if dtype is None:
        dtype = np.arange(start, stop, step * num if num else step).dtype
    chunks = normalize_chunks(chunks, shape=(num,), dtype=dtype)
    chunksize = chunks[0][0]
    numblocks = len(chunks[0])
    # create small array of block numbers
    out = asarray(np.arange(numblocks), chunks=(1,), spec=spec)
    # then map each block to partial arange
    out = map_blocks(_arange, out, dtype=dtype, chunks=chunks, size=chunksize)
    return out


def asarray(obj, /, *, dtype=None, device=None, copy=None, chunks="auto", spec=None):
    a = obj
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    # write to zarr
    chunksize = to_chunksize(normalize_chunks(chunks, shape=a.shape, dtype=dtype))
    name = gensym()
    store, target = new_temp_zarr(a.shape, dtype, chunksize, name=name, spec=spec)
    target[...] = a

    plan = Plan(name, "asarray", target, spec)
    return Array(name, plan, target, target.shape, dtype, chunks)


def ones(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    # write to zarr
    # note that write_empty_chunks=False means no chunks are written to disk, so it is very efficient to create large arrays
    chunksize = to_chunksize(normalize_chunks(chunks, shape=shape, dtype=dtype))
    name = gensym()
    store = new_temp_store(name=name, spec=spec)
    target = zarr.ones(
        shape,
        store=store,
        dtype=dtype,
        chunks=chunksize,
        write_empty_chunks=False,
    )

    plan = Plan(name, "ones", target, spec)
    return Array(name, plan, target, target.shape, target.dtype, chunks)


# Data types

# Use type code from numpy.array_api
from numpy.array_api._dtypes import (  # noqa: F401
    bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

# Data type functions


def result_type(*arrays_and_dtypes):
    # Use numpy.array_api promotion rules (stricter than numpy)
    import numpy.array_api as nxp

    return nxp.result_type(
        *(a.dtype if isinstance(a, Array) else a for a in arrays_and_dtypes)
    )


# Elementwise functions


def add(x1, x2, /):
    return elementwise_binary_operation(x1, x2, np.add, result_type(x1.dtype, x2.dtype))


def divide(x1, x2, /):
    return elementwise_binary_operation(
        x1, x2, np.divide, result_type(x1.dtype, x2.dtype)
    )


def equal(x1, x2, /):
    return elementwise_binary_operation(x1, x2, np.equal, dtype=np.bool)


def negative(x, /):
    return elementwise_unary_operation(x, np.negative, dtype=x.dtype)


# Linear algebra functions


def _matmul(a, b):
    chunk = np.matmul(a, b)
    return chunk[..., np.newaxis, :]


def _sum_wo_cat(a, axis=None, dtype=None):
    if a.shape[axis] == 1:
        return squeeze(a, axis)

    return reduction(a, _chunk_sum, axis=axis, dtype=dtype)


def _chunk_sum(a, axis=None, dtype=None, keepdims=None):
    return np.sum(a, axis=axis, dtype=dtype, keepdims=True)


def matmul(x1, x2, /):
    assert x1.ndim >= 2
    assert x2.ndim >= 2
    assert x1.ndim == x2.ndim

    out_ind = tuple(range(x1.ndim + 1))
    lhs_ind = tuple(range(x1.ndim))
    rhs_ind = tuple(range(x1.ndim - 2)) + (lhs_ind[-1], x1.ndim)

    dtype = result_type(x1, x2)

    out = blockwise(
        _matmul,
        out_ind,
        x1,
        lhs_ind,
        x2,
        rhs_ind,
        adjust_chunks={lhs_ind[-1]: 1},
        dtype=dtype,
    )

    out = _sum_wo_cat(out, axis=-2, dtype=dtype)

    return out


def outer(x1, x2, /):
    return blockwise(np.outer, "ij", x1, "i", x2, "j", dtype=x1.dtype)


# Manipulation functions


def broadcast_to(x, /, shape):
    name = gensym()
    spec = x.plan.spec
    target = primitive_broadcast_to(x.zarray, shape)
    plan = Plan(name, "broadcast_to", target, spec)
    return Array(name, plan, target, target.shape, target.dtype, target.chunks)


def permute_dims(x, /, axes):
    # From dask transpose
    if axes:
        if len(axes) != x.ndim:
            raise ValueError("axes don't match array")
    else:
        axes = tuple(range(x.ndim))[::-1]
    axes = tuple(d + x.ndim if d < 0 else d for d in axes)
    return blockwise(
        np.transpose, axes, x, tuple(range(x.ndim)), dtype=x.dtype, axes=axes
    )


# Statistical functions


def mean(x, /, *, axis=None, keepdims=False):
    # This implementation uses NumPy and Zarr's structured arrays to store a
    # pair of fields needed to keep per-chunk counts and totals for computing
    # the mean. Structured arrays are row-based, so are less efficient than
    # regular arrays, but for a function that reduces the amount of data stored,
    # this is usually OK. An alternative would be to add support for multiple
    # outputs.
    dtype = x.dtype
    combine_dtype = [("n", np.int64), ("total", np.float64)]
    result = reduction(
        x,
        _mean_func,
        combine_func=_mean_combine,
        axis=axis,
        dtype=combine_dtype,
        keepdims=True,
    )
    # TODO: move aggregation into reduction function
    result = map_blocks(_mean_aggregate, result, dtype=dtype)
    if not keepdims:
        result = squeeze(result, axis)
    return result


def _mean_func(a, **kwargs):
    n = numel(a, **kwargs)
    total = np.sum(a, **kwargs)
    return {"n": n, "total": total}


def _mean_combine(a, **kwargs):
    n = np.sum(a["n"], **kwargs)
    total = np.sum(a["total"], **kwargs)
    return {"n": n, "total": total}


def _mean_aggregate(a):
    return np.divide(a["total"], a["n"])


def sum(x, /, *, axis=None, dtype=None, keepdims=False):
    if dtype is None:
        dtype = x.dtype
    return reduction(x, np.sum, axis=axis, dtype=dtype, keepdims=keepdims)


# Utility functions


def all(x, /, *, axis=None, keepdims=False):
    return reduction(x, np.all, axis=axis, dtype=bool, keepdims=keepdims)
