import numpy as np

from cubed.core import elemwise


def add(x1, x2, /):
    return elemwise(np.add, x1, x2)


def divide(x1, x2, /):
    return elemwise(np.divide, x1, x2)


def equal(x1, x2, /):
    return elemwise(np.equal, x1, x2, dtype=np.bool_)


def isfinite(x, /):
    return elemwise(np.isfinite, x, dtype=np.bool_)


def isnan(x, /):
    return elemwise(np.isnan, x, dtype=np.bool_)


def negative(x, /):
    return elemwise(np.negative, x)
