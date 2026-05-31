import numpy as np

from cubed.array_api._numpy_dispatch import implements


@implements(np.isclose)
def _np_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    from cubed.array_api.creation_functions import asarray
    from cubed.array_api.data_type_functions import astype
    from cubed.array_api.dtypes import _boolean_dtypes, _integer_dtypes, float64
    from cubed.array_api.elementwise_functions import abs as xabs
    from cubed.array_api.elementwise_functions import isnan, logical_and, logical_or

    if isinstance(a, np.ndarray):
        a = asarray(a, spec=b.spec)
    if isinstance(b, np.ndarray):
        b = asarray(b, spec=a.spec)

    if a.dtype in _boolean_dtypes or a.dtype in _integer_dtypes:
        a = astype(a, float64)
    if b.dtype in _boolean_dtypes or b.dtype in _integer_dtypes:
        b = astype(b, float64)

    close = xabs(a - b) <= atol + rtol * xabs(b)

    if equal_nan:
        close = logical_or(close, logical_and(isnan(a), isnan(b)))

    return close


@implements(np.result_type)
def _np_result_type(*arrays_and_dtypes):
    from cubed.array_api.array_object import Array

    dtypes = [
        a.dtype if isinstance(a, (Array, np.ndarray)) else a for a in arrays_and_dtypes
    ]
    return np.result_type(*dtypes)


@implements(np.stack)
def _np_stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    from cubed.array_api.array_object import Array
    from cubed.array_api.creation_functions import asarray
    from cubed.array_api.manipulation_functions import stack

    cubed_ref = next(a for a in arrays if isinstance(a, Array))
    converted = [
        asarray(a, spec=cubed_ref.spec, chunks=cubed_ref.chunks)
        if not isinstance(a, Array)
        else a
        for a in arrays
    ]
    return stack(converted, axis=axis)


@implements(np.take)
def _np_take(a, indices, axis=None, out=None, mode="raise"):
    from cubed.array_api.indexing_functions import take

    return take(a, indices, axis=axis)
