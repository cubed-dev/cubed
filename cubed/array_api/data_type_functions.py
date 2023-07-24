from dataclasses import dataclass

import numpy as np
from numpy.array_api._typing import Dtype

from cubed.core import CoreArray, map_blocks

from .dtypes import (
    _all_dtypes,
    _boolean_dtypes,
    _complex_floating_dtypes,
    _integer_dtypes,
    _numeric_dtypes,
    _real_floating_dtypes,
    _result_type,
    _signed_integer_dtypes,
    _unsigned_integer_dtypes,
)


def astype(x, dtype, /, *, copy=True):
    if not copy and dtype == x.dtype:
        return x
    return map_blocks(_astype, x, dtype=dtype, astype_dtype=dtype)


def _astype(a, astype_dtype):
    return a.astype(astype_dtype)


def can_cast(from_, to, /):
    # Copied from numpy.array_api
    # TODO: replace with `nxp.can_cast` when NumPy 1.25 is widely used (e.g. in Xarray)

    if isinstance(from_, CoreArray):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f"{from_=}, but should be an array_api array or dtype")
    if to not in _all_dtypes:
        raise TypeError(f"{to=}, but should be a dtype")
    try:
        # We promote `from_` and `to` together. We then check if the promoted
        # dtype is `to`, which indicates if `from_` can (up)cast to `to`.
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        # _result_type() raises if the dtypes don't promote together
        return False


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: Dtype


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: Dtype


def finfo(type, /):
    # Copied from numpy.array_api
    # TODO: replace with `nxp.finfo(type)` when NumPy 1.25 is widely used (e.g. in Xarray)

    fi = np.finfo(type)
    return finfo_object(
        fi.bits,
        float(fi.eps),
        float(fi.max),
        float(fi.min),
        float(fi.smallest_normal),
        fi.dtype,
    )


def iinfo(type, /):
    # Copied from numpy.array_api
    # TODO: replace with `nxp.iinfo(type)` when NumPy 1.25 is widely used (e.g. in Xarray)

    ii = np.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min, ii.dtype)


def isdtype(dtype, kind):
    # Copied from numpy.array_api
    # TODO: replace with `nxp.isdtype(dtype, kind)` when NumPy 1.25 is widely used (e.g. in Xarray)

    if isinstance(kind, tuple):
        # Disallow nested tuples
        if any(isinstance(k, tuple) for k in kind):
            raise TypeError("'kind' must be a dtype, str, or tuple of dtypes and strs")
        return any(isdtype(dtype, k) for k in kind)
    elif isinstance(kind, str):
        if kind == "bool":
            return dtype in _boolean_dtypes
        elif kind == "signed integer":
            return dtype in _signed_integer_dtypes
        elif kind == "unsigned integer":
            return dtype in _unsigned_integer_dtypes
        elif kind == "integral":
            return dtype in _integer_dtypes
        elif kind == "real floating":
            return dtype in _real_floating_dtypes
        elif kind == "complex floating":
            return dtype in _complex_floating_dtypes
        elif kind == "numeric":
            return dtype in _numeric_dtypes
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    elif kind in _all_dtypes:
        return dtype == kind
    else:
        raise TypeError(
            f"'kind' must be a dtype, str, or tuple of dtypes and strs, not {type(kind).__name__}"
        )


def result_type(*arrays_and_dtypes):
    # Copied from numpy.array_api
    # TODO: replace with `nxp.result_type` when NumPy 1.25 is widely used (e.g. in Xarray)

    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, CoreArray):
            a = a.dtype
        elif isinstance(a, np.ndarray) or a not in _all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = _result_type(t, t2)
        return t
