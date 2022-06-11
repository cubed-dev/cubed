# Use type code from numpy.array_api
from numpy.array_api._dtypes import (  # noqa: F401
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

_numeric_dtypes = (
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
