import numpy as np

# The array implementation used for backend operations.
# This must be compatible with the Python Array API standard, although
# some extra functions are used too (nan functions, take_along_axis),
# which array_api_compat provides, but other Array API implementations
# may not.
import array_api_compat.numpy  # noqa: F401 isort:skip

namespace = array_api_compat.numpy

# These functions to convert to/from backend arrays
# assume that no extra memory is allocated, by using the
# Python buffer protocol.
# See https://data-apis.org/array-api/latest/API_specification/generated/array_api.asarray.html


def backend_array_to_numpy_array(arr):
    return np.asarray(arr)


def numpy_array_to_backend_array(arr, *, dtype=None):
    return namespace.asarray(arr, dtype=dtype)
