# Use type code from numpy.array_api
from numpy.array_api._dtypes import (  # noqa: F401
    _boolean_dtypes,
    _dtype_categories,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

_signed_integer_dtypes = (int8, int16, int32, int64)

_unsigned_integer_dtypes = (uint8, uint16, uint32, uint64)
