import math

from cubed.array_api.dtypes import (
    _numeric_dtypes,
    _real_floating_dtypes,
    _real_numeric_dtypes,
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


def max(x, /, *, axis=None, keepdims=False, use_new_impl=True, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in max")
    return reduction(
        x,
        nxp.max,
        axis=axis,
        dtype=x.dtype,
        use_new_impl=use_new_impl,
        split_every=split_every,
        keepdims=keepdims,
    )


def mean(x, /, *, axis=None, keepdims=False, use_new_impl=True, split_every=None):
    if x.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in mean")
    # This implementation uses NumPy and Zarr's structured arrays to store a
    # pair of fields needed to keep per-chunk counts and totals for computing
    # the mean. Structured arrays are row-based, so are less efficient than
    # regular arrays, but for a function that reduces the amount of data stored,
    # this is usually OK. An alternative would be to add support for multiple
    # outputs.
    dtype = x.dtype
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
        use_new_impl=use_new_impl,
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


def min(x, /, *, axis=None, keepdims=False, use_new_impl=True, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in min")
    return reduction(
        x,
        nxp.min,
        axis=axis,
        dtype=x.dtype,
        use_new_impl=use_new_impl,
        split_every=split_every,
        keepdims=keepdims,
    )


def prod(
    x, /, *, axis=None, dtype=None, keepdims=False, use_new_impl=True, split_every=None
):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")
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
    extra_func_kwargs = dict(dtype=dtype)
    return reduction(
        x,
        nxp.prod,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        use_new_impl=use_new_impl,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )


def sum(
    x, /, *, axis=None, dtype=None, keepdims=False, use_new_impl=True, split_every=None
):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")
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
    extra_func_kwargs = dict(dtype=dtype)
    return reduction(
        x,
        nxp.sum,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        use_new_impl=use_new_impl,
        split_every=split_every,
        extra_func_kwargs=extra_func_kwargs,
    )
