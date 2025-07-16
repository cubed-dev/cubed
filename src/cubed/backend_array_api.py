import os
from importlib import import_module

import numpy as np

# The array implementation used for backend operations is stored in the
# namespace variable, and defaults to array_api_compat.nump, unless it
# is overridden by an environment variable.
# It must be compatible with the Python Array API standard, although
# some extra functions are used too (nan functions, take_along_axis),
# which array_api_compat provides, but other Array API implementations
# may not.

if "CUBED_BACKEND_ARRAY_API_MODULE" in os.environ:
    # This code is based on similar code in array_api_tests
    xp_name = os.environ["CUBED_BACKEND_ARRAY_API_MODULE"]
    _module, _sub = xp_name, None
    if "." in xp_name:
        _module, _sub = xp_name.split(".", 1)
    xp = import_module(_module)
    if _sub:
        try:
            xp = getattr(xp, _sub)
        except AttributeError:
            # _sub may be a submodule that needs to be imported. WE can't
            # do this in every case because some array modules are not
            # submodules that can be imported (like mxnet.nd).
            xp = import_module(xp_name)
    namespace = xp

else:
    import array_api_compat.numpy

    namespace = array_api_compat.numpy


# These functions to convert to/from backend arrays
# assume that no extra memory is allocated, by using the
# Python buffer protocol.
# See https://data-apis.org/array-api/latest/API_specification/generated/array_api.asarray.html


def backend_array_to_numpy_array(arr):
    return np.asarray(arr)


def numpy_array_to_backend_array(arr, *, dtype=None):
    if isinstance(arr, dict):
        return {k: namespace.asarray(v, dtype=dtype) for k, v in arr.items()}
    return namespace.asarray(arr, dtype=dtype)
