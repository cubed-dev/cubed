from cubed.array_api.creation_functions import asarray
from cubed.backend_array_api import namespace as nxp
from cubed.core import reduction


def all(x, /, *, axis=None, keepdims=False):
    if x.size == 0:
        return asarray(True, dtype=x.dtype)
    return reduction(x, nxp.all, axis=axis, dtype=bool, keepdims=keepdims)


def any(x, /, *, axis=None, keepdims=False):
    if x.size == 0:
        return asarray(False, dtype=x.dtype)
    return reduction(x, nxp.any, axis=axis, dtype=bool, keepdims=keepdims)
