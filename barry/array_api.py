from typing import Iterable

import numpy as np
import zarr
from dask.array.core import normalize_chunks

from barry.utils import to_chunksize

from .core import (
    Array,
    Plan,
    blockwise,
    elementwise_binary_operation,
    elementwise_unary_operation,
    gensym,
    new_temp_store,
    new_temp_zarr,
    reduction,
    squeeze,
)

# Creation functions


def asarray(obj, /, *, dtype=None, device=None, copy=None, chunks=None, spec=None):
    a = obj
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    # write to zarr
    chunksize = to_chunksize(normalize_chunks(chunks, a.shape, dtype))
    name = gensym()
    store, target = new_temp_zarr(a.shape, dtype, chunksize, name=name, spec=spec)
    target[:] = a

    plan = Plan(name, "asarray", target, spec)
    return Array(name, plan, store, target.shape, dtype, chunks)


def ones(shape, *, dtype=None, device=None, chunks=None, spec=None):
    # write to zarr
    # note that write_empty_chunks=False means no chunks are written to disk, so it is very efficient to create large arrays
    chunksize = to_chunksize(normalize_chunks(chunks, shape, dtype))
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
    return Array(name, plan, store, target.shape, target.dtype, chunks)


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


def sum(x, /, *, axis=None, dtype=None, keepdims=False):
    if dtype is None:
        dtype = x.dtype
    return reduction(x, np.sum, axis=axis, dtype=dtype, keepdims=keepdims)


# Utility functions


def all(x, /, *, axis=None, keepdims=False):
    return reduction(x, np.all, axis=axis, dtype=bool, keepdims=keepdims)
