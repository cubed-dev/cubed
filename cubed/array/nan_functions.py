import warnings

import numpy as np

import cubed
from cubed.array_api.data_type_functions import isdtype
from cubed.array_api.dtypes import (
    _integer_dtypes,
    _real_numeric_dtypes,
    _upcast_integral_dtypes,
)
from cubed.array_api.elementwise_functions import isnan, sqrt
from cubed.array_api.searching_functions import argmax, argmin, where
from cubed.array_api.statistical_functions import cumulative_prod, cumulative_sum
from cubed.array_api.utility_functions import all, any
from cubed.backend_array_api import namespace as nxp
from cubed.core import reduction

# TODO: refactor once nan functions are standardized:
# https://github.com/data-apis/array-api/issues/621


def nanargmax(x, /, *, axis=None, keepdims=False, split_every=None):
    mask = isnan(x)
    x = where(mask, -cubed.inf, x)
    mask = all(mask, axis=axis, split_every=split_every)
    if any(mask, split_every=split_every):  # eager compute
        raise ValueError("All-NaN slice encountered")
    return argmax(x, axis=axis, keepdims=keepdims)


def nanargmin(x, /, *, axis=None, keepdims=False, split_every=None):
    mask = isnan(x)
    x = where(mask, cubed.inf, x)
    mask = all(mask, axis=axis, split_every=split_every)
    if any(mask, split_every=split_every):  # eager compute
        raise ValueError("All-NaN slice encountered")
    return argmin(x, axis=axis, keepdims=keepdims)


def nancumprod(x, /, *, axis=None, dtype=None):
    x = where(isnan(x), 1, x)
    return cumulative_prod(x, axis=axis, dtype=dtype)


def nancumsum(x, /, *, axis=None, dtype=None):
    x = where(isnan(x), 0, x)
    return cumulative_sum(x, axis=axis, dtype=dtype)


def nanmax(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in nanmax")
    return reduction(
        x,
        _nanmax,
        axis=axis,
        dtype=x.dtype,
        split_every=split_every,
        keepdims=keepdims,
    )


def _nanmax(a, axis=None, keepdims=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return nxp.nanmax(a, axis=axis, keepdims=keepdims)


def nanmean(x, /, *, axis=None, dtype=None, keepdims=False, split_every=None):
    """Compute the arithmetic mean along the specified axis, ignoring NaNs."""
    dtype = dtype or x.dtype
    # TODO(#658): Should these be default dtypes?
    if isdtype(x.dtype, "complex floating"):
        intermediate_dtype = [("n", nxp.int64), ("total", nxp.complex128)]
    else:
        intermediate_dtype = [("n", nxp.int64), ("total", nxp.float64)]
    extra_func_kwargs = dict(dtype=intermediate_dtype)
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
        extra_func_kwargs=extra_func_kwargs,
    )


# note that the array API doesn't have nan functions, so these functions may
# fail at runtime


def _nanmean_func(a, **kwargs):
    dtype = dict(kwargs.pop("dtype"))
    n = _nannumel(a, dtype=dtype["n"], **kwargs)
    total = nxp.nansum(a, dtype=dtype["total"], **kwargs)
    return {"n": n, "total": total}


def _nanmean_combine(a, **kwargs):
    dtype = dict(kwargs.pop("dtype"))
    n = nxp.sum(a["n"], dtype=dtype["n"], **kwargs)
    total = nxp.nansum(a["total"], dtype=dtype["total"], **kwargs)
    return {"n": n, "total": total}


def _nanmean_aggregate(a, **kwargs):
    with np.errstate(divide="ignore", invalid="ignore"):
        return nxp.divide(a["total"], a["n"])


def _nannumel(x, **kwargs):
    """A reduction to count the number of elements, excluding nans"""
    return nxp.sum(~(nxp.isnan(x)), **kwargs)


def nanmin(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in nanmin")
    return reduction(
        x,
        _nanmin,
        axis=axis,
        dtype=x.dtype,
        split_every=split_every,
        keepdims=keepdims,
    )


def _nanmin(a, axis=None, keepdims=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        return nxp.nanmin(a, axis=axis, keepdims=keepdims)


def nanprod(
    x, /, *, axis=None, dtype=None, keepdims=False, split_every=None, device=None
):
    dtype = _upcast_integral_dtypes(
        x,
        dtype,
        allowed_dtypes=(
            "numeric",
            "boolean",
        ),
        fname="nanprod",
        device=device,
    )
    extra_func_kwargs = dict(dtype=dtype)
    return reduction(
        x,
        nxp.nanprod,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def nanstd(x, /, *, axis=None, correction=0.0, keepdims=False, split_every=None):
    return sqrt(
        nanvar(
            x,
            axis=axis,
            correction=correction,
            keepdims=keepdims,
            split_every=split_every,
        )
    )


def nansum(
    x, /, *, axis=None, dtype=None, keepdims=False, split_every=None, device=None
):
    """Return the sum of array elements over a given axis treating NaNs as zero."""
    dtype = _upcast_integral_dtypes(
        x,
        dtype,
        allowed_dtypes=(
            "numeric",
            "boolean",
        ),
        fname="nansum",
        device=device,
    )
    extra_func_kwargs = dict(dtype=dtype)
    return reduction(
        x,
        nxp.nansum,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def nanvar(
    x,
    /,
    *,
    axis=None,
    correction=0.0,
    keepdims=False,
    split_every=None,
):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in nanvar")
    if x.dtype in _integer_dtypes:
        dtype = nxp.__array_namespace_info__().default_dtypes(device=x.device)[
            "real floating"
        ]
    else:
        dtype = x.dtype
    # TODO(#658): Should these be default dtypes?
    intermediate_dtype = [("n", nxp.int64), ("mu", nxp.float64), ("M2", nxp.float64)]
    extra_func_kwargs = dict(dtype=intermediate_dtype, correction=correction)
    return reduction(
        x,
        _nanvar_func,
        combine_func=_nanvar_combine,
        aggregate_func=_nanvar_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def _nanvar_func(a, correction=None, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")

        dtype = dict(kwargs.pop("dtype"))
        n = _nannumel(a, dtype=dtype["n"], **kwargs)
        mu = nxp.nanmean(a, dtype=dtype["mu"], **kwargs)
        M2 = nxp.nansum(nxp.square(a - mu), dtype=dtype["M2"], **kwargs)
        return {"n": n, "mu": mu, "M2": M2}


def _nanvar_combine(a, axis=None, correction=None, **kwargs):
    # _var_combine is called by _partial_reduce which concatenates along the first axis
    axis = axis[0]
    if a["n"].shape[axis] == 1:  # nothing to combine
        return a
    if a["n"].shape[axis] != 2:
        raise ValueError(f"Expected two elements in {axis} axis to combine")

    n_a = nxp.take(a["n"], 0, axis=axis)
    n_b = nxp.take(a["n"], 1, axis=axis)
    mu_a = nxp.take(a["mu"], 0, axis=axis)
    mu_b = nxp.take(a["mu"], 1, axis=axis)
    M2_a = nxp.take(a["M2"], 0, axis=axis)
    M2_b = nxp.take(a["M2"], 1, axis=axis)

    # mu_a and mu_b may be nan - replace with 0
    # so combine works
    mu_a = nxp.where(nxp.isnan(mu_a), 0, mu_a)
    mu_b = nxp.where(nxp.isnan(mu_b), 0, mu_b)

    n_ab = n_a + n_b
    delta = mu_b - mu_a
    mu_ab = (n_a * mu_a + n_b * mu_b) / n_ab
    M2_ab = M2_a + M2_b + delta**2 * n_a * n_b / n_ab

    # mu_ab and M2_ab may be nan if n_ab is 0
    # so replace with 0
    mu_ab = nxp.where(nxp.isnan(mu_ab), 0, mu_ab)
    M2_ab = nxp.where(nxp.isnan(M2_ab), 0, M2_ab)

    n = nxp.expand_dims(n_ab, axis=axis)
    mu = nxp.expand_dims(mu_ab, axis=axis)
    M2 = nxp.expand_dims(M2_ab, axis=axis)

    return {"n": n, "mu": mu, "M2": M2}


def _nanvar_aggregate(a, correction=None, **kwargs):
    with np.errstate(divide="ignore", invalid="ignore"):
        return nxp.divide(a["M2"], a["n"] - correction)
