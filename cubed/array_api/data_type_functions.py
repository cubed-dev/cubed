import numpy.array_api as nxp

from cubed.core import Array, map_blocks


def astype(x, dtype, /, *, copy=True):
    if not copy and dtype == x.dtype:
        return x
    return map_blocks(_astype, x, dtype=dtype, astype_dtype=dtype)


def _astype(a, astype_dtype):
    return a.astype(astype_dtype)


def result_type(*arrays_and_dtypes):
    # Use numpy.array_api promotion rules (stricter than numpy)
    return nxp.result_type(
        *(a.dtype if isinstance(a, Array) else a for a in arrays_and_dtypes)
    )
