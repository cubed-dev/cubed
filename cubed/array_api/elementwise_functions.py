import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import (
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
)
from cubed.core import elemwise


def abs(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    return elemwise(np.abs, x, dtype=x.dtype)


def acos(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return elemwise(np.arccos, x, dtype=x.dtype)


def acosh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return elemwise(np.arccosh, x, dtype=x.dtype)


def add(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    return elemwise(np.add, x1, x2, dtype=result_type(x1, x2))


def asin(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return elemwise(np.arcsin, x, dtype=x.dtype)


def asinh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return elemwise(np.arcsinh, x, dtype=x.dtype)


def atan(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return elemwise(np.arctan, x, dtype=x.dtype)


def atan2(x1, x2, /):
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan2")
    return elemwise(np.arctan2, x1, x2, dtype=result_type(x1, x2))


def atanh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return elemwise(np.arctanh, x, dtype=x.dtype)


def bitwise_and(x1, x2, /):
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    return elemwise(np.bitwise_and, x1, x2, dtype=result_type(x1, x2))


def bitwise_invert(x, /):
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_invert")
    return elemwise(np.invert, x, dtype=x.dtype)


def bitwise_left_shift(x1, x2, /):
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    return elemwise(np.left_shift, x1, x2, dtype=result_type(x1, x2))


def bitwise_or(x1, x2, /):
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    return elemwise(np.bitwise_or, x1, x2, dtype=result_type(x1, x2))


def bitwise_right_shift(x1, x2, /):
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_right_shift")
    return elemwise(np.right_shift, x1, x2, dtype=result_type(x1, x2))


def bitwise_xor(x1, x2, /):
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    return elemwise(np.bitwise_xor, x1, x2, dtype=result_type(x1, x2))


def ceil(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in ceil")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return elemwise(np.ceil, x, dtype=x.dtype)


def cos(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return elemwise(np.cos, x, dtype=x.dtype)


def cosh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return elemwise(np.cosh, x, dtype=x.dtype)


def divide(x1, x2, /):
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    return elemwise(np.divide, x1, x2, dtype=result_type(x1, x2))


def exp(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return elemwise(np.exp, x, dtype=x.dtype)


def expm1(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in expm1")
    return elemwise(np.expm1, x, dtype=x.dtype)


def equal(x1, x2, /):
    return elemwise(np.equal, x1, x2, dtype=np.bool_)


def floor(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return elemwise(np.floor, x, dtype=x.dtype)


def floor_divide(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor_divide")
    return elemwise(np.floor_divide, x1, x2, dtype=result_type(x1, x2))


def greater(x1, x2, /):
    return elemwise(np.greater, x1, x2, dtype=np.bool_)


def greater_equal(x1, x2, /):
    return elemwise(np.greater_equal, x1, x2, dtype=np.bool_)


def isfinite(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return elemwise(np.isfinite, x, dtype=np.bool_)


def isinf(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return elemwise(np.isinf, x, dtype=np.bool_)


def isnan(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return elemwise(np.isnan, x, dtype=np.bool_)


def less(x1, x2, /):
    return elemwise(np.less, x1, x2, dtype=np.bool_)


def less_equal(x1, x2, /):
    return elemwise(np.less_equal, x1, x2, dtype=np.bool_)


def log(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return elemwise(np.log, x, dtype=x.dtype)


def log1p(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log1p")
    return elemwise(np.log1p, x, dtype=x.dtype)


def log2(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log2")
    return elemwise(np.log2, x, dtype=x.dtype)


def log10(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log10")
    return elemwise(np.log10, x, dtype=x.dtype)


def logaddexp(x1, x2, /):
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in logaddexp")
    return elemwise(np.logaddexp, x1, x2, dtype=result_type(x1, x2))


def logical_and(x1, x2, /):
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    return elemwise(np.logical_and, x1, x2, dtype=np.bool_)


def logical_not(x, /):
    if x.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_not")
    return elemwise(np.logical_not, x, dtype=np.bool_)


def logical_or(x1, x2, /):
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    return elemwise(np.logical_or, x1, x2, dtype=np.bool_)


def logical_xor(x1, x2, /):
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    return elemwise(np.logical_xor, x1, x2, dtype=np.bool_)


def multiply(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    return elemwise(np.multiply, x1, x2, dtype=result_type(x1, x2))


def negative(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in negative")
    return elemwise(np.negative, x, dtype=x.dtype)


def not_equal(x1, x2, /):
    return elemwise(np.not_equal, x1, x2, dtype=np.bool_)


def positive(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in positive")
    return elemwise(np.positive, x, dtype=x.dtype)


def pow(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in pow")
    return elemwise(np.power, x1, x2, dtype=result_type(x1, x2))


def remainder(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in remainder")
    return elemwise(np.remainder, x1, x2, dtype=result_type(x1, x2))


def round(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    return elemwise(np.round, x, dtype=x.dtype)


def sign(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return elemwise(np.sign, x, dtype=x.dtype)


def sin(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return elemwise(np.sin, x, dtype=x.dtype)


def sinh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return elemwise(np.sinh, x, dtype=x.dtype)


def sqrt(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return elemwise(np.sqrt, x, dtype=x.dtype)


def square(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in square")
    return elemwise(np.square, x, dtype=x.dtype)


def subtract(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    return elemwise(np.subtract, x1, x2, dtype=result_type(x1, x2))


def tan(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return elemwise(np.tan, x, dtype=x.dtype)


def tanh(x, /):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return elemwise(np.tanh, x, dtype=x.dtype)


def trunc(x, /):
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in trunc")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of trunc is the same as the input
        return x
    return elemwise(np.trunc, x, dtype=x.dtype)
