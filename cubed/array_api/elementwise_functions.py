import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.core import elemwise


def add(x1, x2, /):
    return elemwise(np.add, x1, x2, dtype=result_type(x1, x2))


def divide(x1, x2, /):
    return elemwise(np.divide, x1, x2, dtype=result_type(x1, x2))


def equal(x1, x2, /):
    return elemwise(np.equal, x1, x2, dtype=np.bool_)


def isfinite(x, /):
    return elemwise(np.isfinite, x, dtype=np.bool_)


def isinf(x, /):
    return elemwise(np.isinf, x, dtype=np.bool_)


def isnan(x, /):
    return elemwise(np.isnan, x, dtype=np.bool_)


def logical_and(x1, x2, /):
    return elemwise(np.logical_and, x1, x2, dtype=np.bool_)


def logical_or(x1, x2, /):
    return elemwise(np.logical_or, x1, x2, dtype=np.bool_)


def negative(x, /):
    return elemwise(np.negative, x, dtype=x.dtype)
