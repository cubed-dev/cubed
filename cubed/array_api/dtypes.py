# Copied from numpy.array_api
from cubed.backend_array_api import namespace as nxp

int8 = nxp.int8
int16 = nxp.int16
int32 = nxp.int32
int64 = nxp.int64
uint8 = nxp.uint8
uint16 = nxp.uint16
uint32 = nxp.uint32
uint64 = nxp.uint64
float32 = nxp.float32
float64 = nxp.float64
complex64 = nxp.complex64
complex128 = nxp.complex128
bool = nxp.bool

_all_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    bool,
)
_boolean_dtypes = (bool,)
_real_floating_dtypes = (float32, float64)
_floating_dtypes = (float32, float64, complex64, complex128)
_complex_floating_dtypes = (complex64, complex128)
_integer_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
_signed_integer_dtypes = (int8, int16, int32, int64)
_unsigned_integer_dtypes = (uint8, uint16, uint32, uint64)
_integer_or_boolean_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
_real_numeric_dtypes = (
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
    complex64,
    complex128,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

_dtype_categories = {
    "all": _all_dtypes,
    "real numeric": _real_numeric_dtypes,
    "numeric": _numeric_dtypes,
    "integer": _integer_dtypes,
    "integer or boolean": _integer_or_boolean_dtypes,
    "boolean": _boolean_dtypes,
    "real floating-point": _floating_dtypes,
    "complex floating-point": _complex_floating_dtypes,
    "floating-point": _floating_dtypes,
}
