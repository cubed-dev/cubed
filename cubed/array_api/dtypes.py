# Copied from numpy.array_api
from cubed.array_api.inspection import __array_namespace_info__
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


# A Cubed-specific utility.
def _upcast_integral_dtypes(
    x, dtype=None, *, allowed_dtypes=("numeric",), fname=None, device=None
):
    """Ensure the input dtype is allowed. If it's None, provide a good default dtype."""
    dtypes = __array_namespace_info__().default_dtypes(device=device)

    # Validate.
    is_invalid = all(x.dtype not in _dtype_categories[a] for a in allowed_dtypes)
    if is_invalid:
        errmsg = f"Only {' or '.join(allowed_dtypes)} dtypes are allowed"
        if fname:
            errmsg += f" in {fname}"
        raise TypeError(errmsg)

    # Choose a good default dtype, when None
    if dtype is None:
        if x.dtype in _boolean_dtypes:
            dtype = dtypes["integral"]
        elif x.dtype in _signed_integer_dtypes:
            dtype = dtypes["integral"]
        elif x.dtype in _unsigned_integer_dtypes:
            # Type arithmetic to produce an unsigned integer dtype at the same default precision.
            default_bits = nxp.iinfo(dtypes["integral"]).bits
            dtype = nxp.dtype(f"u{default_bits // 8}")
        else:
            dtype = x.dtype

    return dtype


def _promote_scalars(x1, x2, op):
    """Promote at most one of x1 or x2 to an array from a Python scalar"""
    x1_is_scalar = isinstance(x1, (int, float, complex, bool))
    x2_is_scalar = isinstance(x2, (int, float, complex, bool))
    if x1_is_scalar and x2_is_scalar:
        raise TypeError(f"At least one of x1 and x2 must be an array in {op}")
    elif x1_is_scalar:
        x1 = x2._promote_scalar(x1)
    elif x2_is_scalar:
        x2 = x1._promote_scalar(x2)
    return x1, x2
