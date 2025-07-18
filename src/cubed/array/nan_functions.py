import numpy as np

from cubed.array_api.dtypes import _upcast_integral_dtypes
from cubed.backend_array_api import namespace as nxp
from cubed.core import reduction

# TODO: refactor once nan functions are standardized:
# https://github.com/data-apis/array-api/issues/621


def nanmean(x, /, *, axis=None, dtype=None, keepdims=False, split_every=None):
    """Compute the arithmetic mean along the specified axis, ignoring NaNs."""
    dtype = dtype or x.dtype
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
    x, /, *, axis=None, dtype=None, keepdims=False, split_every=None, device=None
):
    """Return the sum of array elements over a given axis treating NaNs as zero."""
    dtype = _upcast_integral_dtypes(
        x, dtype, allowed_dtypes=("numeric",), fname="nansum", device=device
    )
    return reduction(
        x,
        nxp.nansum,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
    )
