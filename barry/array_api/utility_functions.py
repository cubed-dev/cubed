import numpy as np

from barry.core import reduction


def all(x, /, *, axis=None, keepdims=False):
    return reduction(x, np.all, axis=axis, dtype=bool, keepdims=keepdims)
