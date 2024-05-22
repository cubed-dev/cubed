import numpy as np

from cubed.array_api.dtypes import (
    _numeric_dtypes,
    _signed_integer_dtypes,
    _unsigned_integer_dtypes,
    complex64,
    complex128,
    float32,
    float64,
    int64,
    uint64,
)
from cubed.backend_array_api import namespace as nxp
from cubed.core import reduction

# TODO: refactor once nan functions are standardized:
# https://github.com/data-apis/array-api/issues/621


def nanmean(x, /, *, axis=None, keepdims=False, use_new_impl=True, split_every=None):
    """Compute the arithmetic mean along the specified axis, ignoring NaNs."""
    dtype = x.dtype
    intermediate_dtype = [("n", nxp.int64), ("total", nxp.float64)]
    return reduction(
        x,
        _nanmean_func,
        combine_func=_nanmean_combine,
        aggregate_func=_nanmean_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
        use_new_impl=use_new_impl,
        split_every=split_every,
    )


# note that the array API doesn't have nansum or nanmean, so these functions may fail


def _nanmean_func(a, **kwargs):
    n = _nannumel(a, **kwargs)
    total = nxp.nansum(a, **kwargs)
    return {"n": n, "total": total}


def _nanmean_combine(a, **kwargs):
    n = nxp.nansum(a["n"], **kwargs)
    total = nxp.nansum(a["total"], **kwargs)
    return {"n": n, "total": total}


def _nanmean_aggregate(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return nxp.divide(a["total"], a["n"])


def _nannumel(x, **kwargs):
    """A reduction to count the number of elements, excluding nans"""
    return nxp.sum(~(nxp.isnan(x)), **kwargs)


def nansum(
    x, /, *, axis=None, dtype=None, keepdims=False, use_new_impl=True, split_every=None
):
    """Return the sum of array elements over a given axis treating NaNs as zero."""
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in nansum")
    if dtype is None:
        if x.dtype in _signed_integer_dtypes:
            dtype = int64
        elif x.dtype in _unsigned_integer_dtypes:
            dtype = uint64
        elif x.dtype == float32:
            dtype = float64
        elif x.dtype == complex64:
            dtype = complex128
        else:
            dtype = x.dtype
    return reduction(
        x,
        nxp.nansum,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        use_new_impl=use_new_impl,
        split_every=split_every,
    )
