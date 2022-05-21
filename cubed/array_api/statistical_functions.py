import numpy as np
from dask.array.reductions import numel

from cubed.core import map_blocks, reduction, squeeze


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
