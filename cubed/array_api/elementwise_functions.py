import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import _floating_dtypes, _numeric_dtypes
from cubed.core import elemwise


def add(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    return elemwise(np.add, x1, x2, dtype=result_type(x1, x2))


def divide(x1, x2, /):
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    return elemwise(np.divide, x1, x2, dtype=result_type(x1, x2))


def equal(x1, x2, /):
    return elemwise(np.equal, x1, x2, dtype=np.bool_)


def floor_divide(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor_divide")
    return elemwise(np.floor_divide, x1, x2, dtype=result_type(x1, x2))


def greater(x1, x2, /):
    return elemwise(np.greater, x1, x2, dtype=np.bool_)


def greater_equal(x1, x2, /):
    return elemwise(np.greater_equal, x1, x2, dtype=np.bool_)


def isfinite(x, /):
    return elemwise(np.isfinite, x, dtype=np.bool_)


def isinf(x, /):
    return elemwise(np.isinf, x, dtype=np.bool_)


def isnan(x, /):
    return elemwise(np.isnan, x, dtype=np.bool_)


def less(x1, x2, /):
    return elemwise(np.less, x1, x2, dtype=np.bool_)


def less_equal(x1, x2, /):
    return elemwise(np.less_equal, x1, x2, dtype=np.bool_)


def logical_and(x1, x2, /):
    return elemwise(np.logical_and, x1, x2, dtype=np.bool_)


def logical_or(x1, x2, /):
    return elemwise(np.logical_or, x1, x2, dtype=np.bool_)


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


def subtract(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    return elemwise(np.subtract, x1, x2, dtype=result_type(x1, x2))
