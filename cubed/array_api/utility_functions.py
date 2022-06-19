import numpy as np

from cubed.array_api.creation_functions import asarray
from cubed.core import reduction


def all(x, /, *, axis=None, keepdims=False):
    if x.size == 0:
        return asarray(True, dtype=x.dtype)
    return reduction(x, np.all, axis=axis, dtype=bool, keepdims=keepdims)


def any(x, /, *, axis=None, keepdims=False):
    if x.size == 0:
        return asarray(False, dtype=x.dtype)
    return reduction(x, np.any, axis=axis, dtype=bool, keepdims=keepdims)
