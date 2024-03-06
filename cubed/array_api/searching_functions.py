from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import _real_numeric_dtypes
from cubed.array_api.manipulation_functions import reshape
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import arg_reduction, elemwise


def argmax(x, /, *, axis=None, keepdims=False, use_new_impl=True, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmax")
    if axis is None:
        x = reshape(x, (-1,))
        axis = 0
        keepdims = False
    return arg_reduction(
        x,
        nxp.argmax,
        axis=axis,
        keepdims=keepdims,
        use_new_impl=use_new_impl,
        split_every=split_every,
    )


def argmin(x, /, *, axis=None, keepdims=False, use_new_impl=True, split_every=None):
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmin")
    if axis is None:
        x = reshape(x, (-1,))
        axis = 0
        keepdims = False
    return arg_reduction(
        x,
        nxp.argmin,
        axis=axis,
        keepdims=keepdims,
        use_new_impl=use_new_impl,
        split_every=split_every,
    )


def where(condition, x1, x2, /):
    dtype = result_type(x1, x2)
    return elemwise(nxp.where, condition, x1, x2, dtype=dtype)
