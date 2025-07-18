from cubed.backend_array_api import namespace as nxp
from cubed.core import CoreArray, map_blocks


def astype(x, dtype, /, *, copy=True, device=None):
    if not copy and dtype == x.dtype:
        return x
    return map_blocks(_astype, x, dtype=dtype, astype_dtype=dtype)


def _astype(a, astype_dtype):
    return nxp.astype(a, astype_dtype)


def can_cast(from_, to, /):
    if isinstance(from_, CoreArray):
        from_ = from_.dtype
    return nxp.can_cast(from_, to)


def finfo(type, /):
    return nxp.finfo(type)


def iinfo(type, /):
    return nxp.iinfo(type)


def isdtype(dtype, kind):
    return nxp.isdtype(dtype, kind)


def result_type(*arrays_and_dtypes):
    return nxp.result_type(
        *(a.dtype if isinstance(a, CoreArray) else a for a in arrays_and_dtypes)
    )
