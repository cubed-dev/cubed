import math

from cubed.array_api.dtypes import (
    _real_floating_dtypes,
    _real_numeric_dtypes,
    _upcast_integral_dtypes,
)
from cubed.array_api.elementwise_functions import sqrt
from cubed.backend_array_api import namespace as nxp
from cubed.core import reduction


def max(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in max")
    return reduction(
        x,
        nxp.max,
        axis=axis,
        dtype=x.dtype,
        split_every=split_every,
        keepdims=keepdims,
    )


def mean(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in mean")
    # This implementation uses a Zarr group of two arrays to store a
    # pair of fields needed to keep per-chunk counts and totals for computing
    # the mean.
    dtype = x.dtype
    # TODO(#658): Should these be default dtypes?
    intermediate_dtype = [("n", nxp.int64), ("total", nxp.float64)]
    extra_func_kwargs = dict(dtype=intermediate_dtype)
    return reduction(
        x,
        _mean_func,
        combine_func=_mean_combine,
        aggregate_func=_mean_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def _mean_func(a, **kwargs):
    dtype = dict(kwargs.pop("dtype"))
    n = _numel(a, dtype=dtype["n"], **kwargs)
    total = nxp.sum(a, dtype=dtype["total"], **kwargs)
    return {"n": n, "total": total}


def _mean_combine(a, **kwargs):
    dtype = dict(kwargs.pop("dtype"))
    n = nxp.sum(a["n"], dtype=dtype["n"], **kwargs)
    total = nxp.sum(a["total"], dtype=dtype["total"], **kwargs)
    return {"n": n, "total": total}


def _mean_aggregate(a, **kwargs):
    return nxp.divide(a["total"], a["n"])


# based on dask
def _numel(x, **kwargs):
    """
    A reduction to count the number of elements.
    """
    shape = x.shape
    keepdims = kwargs.get("keepdims", False)
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", nxp.float64)

    if axis is None:
        prod = nxp.prod(shape, dtype=dtype)
        if keepdims is False:
            return prod

        return nxp.full(shape=(1,) * len(shape), fill_value=prod, dtype=dtype)

    if not isinstance(axis, (tuple, list)):
        axis = [axis]

    prod = math.prod(shape[dim] for dim in axis)
    if keepdims is True:
        new_shape = tuple(
            shape[dim] if dim not in axis else 1 for dim in range(len(shape))
        )
    else:
        new_shape = tuple(shape[dim] for dim in range(len(shape)) if dim not in axis)

    return nxp.broadcast_to(nxp.asarray(prod, dtype=dtype), new_shape)


def min(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")
    return reduction(
        x,
        nxp.min,
        axis=axis,
        dtype=x.dtype,
        split_every=split_every,
        keepdims=keepdims,
    )


def prod(x, /, *, axis=None, dtype=None, keepdims=False, split_every=None, device=None):
    dtype = _upcast_integral_dtypes(
        x,
        dtype,
        allowed_dtypes=(
            "numeric",
            "boolean",
        ),
        fname="prod",
        device=device,
    )
    extra_func_kwargs = dict(dtype=dtype)
    return reduction(
        x,
        nxp.prod,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def std(x, /, *, axis=None, correction=0.0, keepdims=False, split_every=None):
    return sqrt(
        var(
            x,
            axis=axis,
            correction=correction,
            keepdims=keepdims,
            split_every=split_every,
        )
    )


def sum(x, /, *, axis=None, dtype=None, keepdims=False, split_every=None, device=None):
    dtype = _upcast_integral_dtypes(
        x,
        dtype,
        allowed_dtypes=(
            "numeric",
            "boolean",
        ),
        fname="sum",
        device=device,
    )
    extra_func_kwargs = dict(dtype=dtype)
    return reduction(
        x,
        nxp.sum,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def var(
    x,
    /,
    *,
    axis=None,
    correction=0.0,
    keepdims=False,
    split_every=None,
):
    # This implementation follows https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in var")
    dtype = x.dtype
    # TODO(#658): Should these be default dtypes?
    intermediate_dtype = [("n", nxp.int64), ("mu", nxp.float64), ("M2", nxp.float64)]
    extra_func_kwargs = dict(dtype=intermediate_dtype, correction=correction)
    return reduction(
        x,
        _var_func,
        combine_func=_var_combine,
        aggregate_func=_var_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def _var_func(a, correction=None, **kwargs):
    dtype = dict(kwargs.pop("dtype"))
    n = _numel(a, dtype=dtype["n"], **kwargs)
    mu = nxp.mean(a, dtype=dtype["mu"], **kwargs)
    M2 = nxp.sum(nxp.square(a - mu), dtype=dtype["M2"], **kwargs)
    return {"n": n, "mu": mu, "M2": M2}


def _var_combine(a, axis=None, correction=None, **kwargs):
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

    n_ab = n_a + n_b
    delta = mu_b - mu_a
    mu_ab = (n_a * mu_a + n_b * mu_b) / n_ab
    M2_ab = M2_a + M2_b + delta**2 * n_a * n_b / n_ab

    n = nxp.expand_dims(n_ab, axis=axis)
    mu = nxp.expand_dims(mu_ab, axis=axis)
    M2 = nxp.expand_dims(M2_ab, axis=axis)

    return {"n": n, "mu": mu, "M2": M2}


def _var_aggregate(a, correction=None, **kwargs):
    return nxp.divide(a["M2"], a["n"] - correction)
