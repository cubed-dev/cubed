from cubed.array_api.creation_functions import asarray
from cubed.array_api.dtypes import (
    _signed_integer_dtypes,
    _unsigned_integer_dtypes,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)
from cubed.backend_array_api import namespace as nxp, namespace, PRECISION
from cubed.core import reduction


def all(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.size == 0:
        return asarray(True, dtype=x.dtype)
    return reduction(
        x,
        nxp.all,
        axis=axis,
        dtype=bool,
        keepdims=keepdims,
        split_every=split_every,
    )


def any(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.size == 0:
        return asarray(False, dtype=x.dtype)
    return reduction(
        x,
        nxp.any,
        axis=axis,
        dtype=bool,
        keepdims=keepdims,
        split_every=split_every,
    )


def operator_default_dtype(x: namespace.ndarray) -> namespace.dtype:
    """Derive the correct default data type for operators."""
    if x.dtype in _signed_integer_dtypes:
        dtype = int64 if PRECISION == 64 else int32
    elif x.dtype in _unsigned_integer_dtypes:
        dtype = uint64 if PRECISION == 64 else uint32
    elif x.dtype == float32 and PRECISION == 64:
        dtype = float64
    elif x.dtype == complex64 and PRECISION == 64:
        dtype = complex128
    else:
        dtype = x.dtype

    return dtype
