import numpy as np

from cubed.core import reduction


def all(x, /, *, axis=None, keepdims=False):
    return reduction(x, np.all, axis=axis, dtype=bool, keepdims=keepdims)


def any(x, /, *, axis=None, keepdims=False):
    return reduction(x, np.any, axis=axis, dtype=bool, keepdims=keepdims)
