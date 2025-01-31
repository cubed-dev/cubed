from cubed.array_api.array_object import Array
from cubed.array_api.creation_functions import asarray
from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import (
    _boolean_dtypes,
    _complex_floating_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    _promote_scalars,
    _real_floating_dtypes,
    _real_numeric_dtypes,
    complex64,
    complex128,
    float32,
    float64,
)
from cubed.backend_array_api import namespace as nxp
from cubed.core import elemwise


def abs(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    if x.dtype == complex64:
        dtype = float32
    elif x.dtype == complex128:
        dtype = float64
    else:
        dtype = x.dtype
    return elemwise(nxp.abs, x, dtype=dtype)


def acos(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return elemwise(nxp.acos, x, dtype=x.dtype)


def acosh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return elemwise(nxp.acosh, x, dtype=x.dtype)


def add(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "add")
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    return elemwise(nxp.add, x1, x2, dtype=result_type(x1, x2))


def asin(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return elemwise(nxp.asin, x, dtype=x.dtype)


def asinh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return elemwise(nxp.asinh, x, dtype=x.dtype)


def atan(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return elemwise(nxp.atan, x, dtype=x.dtype)


def atan2(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "atan2")
    if x1.dtype not in _real_floating_dtypes or x2.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in atan2")
    return elemwise(nxp.atan2, x1, x2, dtype=result_type(x1, x2))


def atanh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return elemwise(nxp.atanh, x, dtype=x.dtype)


def bitwise_and(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "bitwise_and")
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    return elemwise(nxp.bitwise_and, x1, x2, dtype=result_type(x1, x2))


def bitwise_invert(x, /):
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_invert")
    return elemwise(nxp.bitwise_invert, x, dtype=x.dtype)


def bitwise_left_shift(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "bitwise_left_shift")
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    return elemwise(nxp.bitwise_left_shift, x1, x2, dtype=result_type(x1, x2))


def bitwise_or(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "bitwise_or")
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    return elemwise(nxp.bitwise_or, x1, x2, dtype=result_type(x1, x2))


def bitwise_right_shift(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "bitwise_right_shift")
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_right_shift")
    return elemwise(nxp.bitwise_right_shift, x1, x2, dtype=result_type(x1, x2))


def bitwise_xor(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "bitwise_xor")
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    return elemwise(nxp.bitwise_xor, x1, x2, dtype=result_type(x1, x2))


def ceil(x, /):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in ceil")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return elemwise(nxp.ceil, x, dtype=x.dtype)


def clip(x, /, min=None, max=None):
    if (
        x.dtype not in _real_numeric_dtypes
        or isinstance(min, Array)
        and min.dtype not in _real_numeric_dtypes
        or isinstance(max, Array)
        and max.dtype not in _real_numeric_dtypes
    ):
        raise TypeError("Only real numeric dtypes are allowed in clip")
    if not isinstance(min, (int, float, Array, type(None))):
        raise TypeError("min must be an None, int, float, or an array")
    if not isinstance(max, (int, float, Array, type(None))):
        raise TypeError("max must be an None, int, float, or an array")

    if min is max is None:
        return x
    elif min is not None and max is None:
        min = asarray(min, spec=x.spec)
        return elemwise(nxp.clip, x, min, dtype=x.dtype)
    elif min is None and max is not None:

        def clip_max(x_, max_):
            return nxp.clip(x_, max=max_)

        max = asarray(max, spec=x.spec)
        return elemwise(clip_max, x, max, dtype=x.dtype)
    else:  # min is not None and max is not None
        min = asarray(min, spec=x.spec)
        max = asarray(max, spec=x.spec)
        return elemwise(nxp.clip, x, min, max, dtype=x.dtype)


def conj(x, /):
    if x.dtype not in _complex_floating_dtypes:
        raise TypeError("Only complex floating-point dtypes are allowed in conj")
    return elemwise(nxp.conj, x, dtype=x.dtype)


def copysign(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "copysign")
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in copysign")
    return elemwise(nxp.copysign, x1, x2, dtype=result_type(x1, x2))


def cos(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return elemwise(nxp.cos, x, dtype=x.dtype)


def cosh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return elemwise(nxp.cosh, x, dtype=x.dtype)


def divide(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "divide")
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    return elemwise(nxp.divide, x1, x2, dtype=result_type(x1, x2))


def exp(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return elemwise(nxp.exp, x, dtype=x.dtype)


def expm1(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in expm1")
    return elemwise(nxp.expm1, x, dtype=x.dtype)


def equal(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "equal")
    return elemwise(nxp.equal, x1, x2, dtype=nxp.bool)


def floor(x, /):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in floor")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return elemwise(nxp.floor, x, dtype=x.dtype)


def floor_divide(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "floor_divide")
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in floor_divide")
    return elemwise(nxp.floor_divide, x1, x2, dtype=result_type(x1, x2))


def greater(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "greater")
    return elemwise(nxp.greater, x1, x2, dtype=nxp.bool)


def greater_equal(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "greater_equal")
    return elemwise(nxp.greater_equal, x1, x2, dtype=nxp.bool)


def hypot(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "hypot")
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in hypot")
    return elemwise(nxp.hypot, x1, x2, dtype=result_type(x1, x2))


def imag(x, /):
    if x.dtype == complex64:
        dtype = float32
    elif x.dtype == complex128:
        dtype = float64
    else:
        raise TypeError("Only complex floating-point dtypes are allowed in imag")
    return elemwise(nxp.imag, x, dtype=dtype)


def isfinite(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return elemwise(nxp.isfinite, x, dtype=nxp.bool)


def isinf(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return elemwise(nxp.isinf, x, dtype=nxp.bool)


def isnan(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return elemwise(nxp.isnan, x, dtype=nxp.bool)


def less(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "less")
    return elemwise(nxp.less, x1, x2, dtype=nxp.bool)


def less_equal(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "less_equal")
    return elemwise(nxp.less_equal, x1, x2, dtype=nxp.bool)


def log(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return elemwise(nxp.log, x, dtype=x.dtype)


def log1p(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log1p")
    return elemwise(nxp.log1p, x, dtype=x.dtype)


def log2(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log2")
    return elemwise(nxp.log2, x, dtype=x.dtype)


def log10(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log10")
    return elemwise(nxp.log10, x, dtype=x.dtype)


def logaddexp(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "logaddexp")
    if x1.dtype not in _real_floating_dtypes or x2.dtype not in _real_floating_dtypes:
        raise TypeError("Only real floating-point dtypes are allowed in logaddexp")
    return elemwise(nxp.logaddexp, x1, x2, dtype=result_type(x1, x2))


def logical_and(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "logical_and")
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    return elemwise(nxp.logical_and, x1, x2, dtype=nxp.bool)


def logical_not(x, /):
    if x.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_not")
    return elemwise(nxp.logical_not, x, dtype=nxp.bool)


def logical_or(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "logical_or")
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    return elemwise(nxp.logical_or, x1, x2, dtype=nxp.bool)


def logical_xor(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "logical_xor")
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    return elemwise(nxp.logical_xor, x1, x2, dtype=nxp.bool)


def maximum(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "maximum")
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in maximum")
    return elemwise(nxp.maximum, x1, x2, dtype=result_type(x1, x2))


def minimum(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "minimum")
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in minimum")
    return elemwise(nxp.minimum, x1, x2, dtype=result_type(x1, x2))


def multiply(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "multiply")
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    return elemwise(nxp.multiply, x1, x2, dtype=result_type(x1, x2))


def negative(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in negative")
    return elemwise(nxp.negative, x, dtype=x.dtype)


def not_equal(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "not_equal")
    return elemwise(nxp.not_equal, x1, x2, dtype=nxp.bool)


def positive(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in positive")
    return elemwise(nxp.positive, x, dtype=x.dtype)


def pow(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "pow")
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in pow")
    return elemwise(nxp.pow, x1, x2, dtype=result_type(x1, x2))


def real(x, /):
    if x.dtype == complex64:
        dtype = float32
    elif x.dtype == complex128:
        dtype = float64
    else:
        raise TypeError("Only complex floating-point dtypes are allowed in real")
    return elemwise(nxp.real, x, dtype=dtype)


def remainder(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "remainder")
    if x1.dtype not in _real_numeric_dtypes or x2.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in remainder")
    return elemwise(nxp.remainder, x1, x2, dtype=result_type(x1, x2))


def round(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    return elemwise(nxp.round, x, dtype=x.dtype)


def sign(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return elemwise(nxp.sign, x, dtype=x.dtype)


def signbit(x, /):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in signbit")
    return elemwise(nxp.signbit, x, dtype=nxp.bool)


def sin(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return elemwise(nxp.sin, x, dtype=x.dtype)


def sinh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return elemwise(nxp.sinh, x, dtype=x.dtype)


def sqrt(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return elemwise(nxp.sqrt, x, dtype=x.dtype)


def square(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in square")
    return elemwise(nxp.square, x, dtype=x.dtype)


def subtract(x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "subtract")
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    return elemwise(nxp.subtract, x1, x2, dtype=result_type(x1, x2))


def tan(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return elemwise(nxp.tan, x, dtype=x.dtype)


def tanh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return elemwise(nxp.tanh, x, dtype=x.dtype)


def trunc(x, /):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in trunc")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of trunc is the same as the input
        return x
    return elemwise(nxp.trunc, x, dtype=x.dtype)
