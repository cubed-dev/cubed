from numbers import Integral
from typing import Iterable

import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import _numeric_dtypes
from cubed.array_api.manipulation_functions import expand_dims
from cubed.core import blockwise, reduction, squeeze


def matmul(x1, x2, /):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in matmul")

    if x1.ndim == 0 or x2.ndim == 0:
        raise ValueError("matmul does not support 0-dimensional arrays.")

    x1_is_1d = False
    if x1.ndim == 1:
        x1_is_1d = True
        x1 = expand_dims(x1, axis=0)

    x2_is_1d = False
    if x2.ndim == 1:
        x2_is_1d = True
        x2 = expand_dims(x2, axis=-1)

    if x1.ndim < x2.ndim:
        x1 = expand_dims(x1, tuple(range(x2.ndim - x1.ndim)))
    elif x1.ndim > x2.ndim:
        x2 = expand_dims(x2, tuple(range(x1.ndim - x2.ndim)))

    out_ind = tuple(range(x1.ndim + 1))
    x1_ind = tuple(range(x1.ndim))
    x2_ind = tuple(range(x1.ndim - 2)) + (x1_ind[-1], x1.ndim)

    dtype = result_type(x1, x2)

    out = blockwise(
        _matmul,
        out_ind,
        x1,
        x1_ind,
        x2,
        x2_ind,
        adjust_chunks={x1_ind[-1]: 1},
        dtype=dtype,
    )

    out = _sum_wo_cat(out, axis=-2, dtype=dtype)

    if x1_is_1d:
        out = squeeze(out, -2)
    if x2_is_1d:
        out = squeeze(out, -1)

    return out


def _matmul(a, b):
    chunk = np.matmul(a, b)
    return chunk[..., np.newaxis, :]


def _sum_wo_cat(a, axis=None, dtype=None):
    if a.shape[axis] == 1:
        return squeeze(a, axis)

    return reduction(a, _chunk_sum, axis=axis, dtype=dtype)


def _chunk_sum(a, axis=None, dtype=None, keepdims=None):
    return np.sum(a, axis=axis, dtype=dtype, keepdims=True)


def matrix_transpose(x, /):
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")

    from cubed.array_api.manipulation_functions import permute_dims

    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]  # swap last two axes
    return permute_dims(x, axes)


def outer(x1, x2, /):
    return blockwise(np.outer, "ij", x1, "i", x2, "j", dtype=x1.dtype)


def tensordot(x1, x2, /, *, axes=2):

    from cubed.array_api.statistical_functions import sum

    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in tensordot")

    # based on dask

    if isinstance(axes, Iterable):
        x1_axes, x2_axes = axes
    else:
        x1_axes = tuple(range(x1.ndim - axes, x1.ndim))
        x2_axes = tuple(range(0, axes))
    if isinstance(x1_axes, Integral):
        x1_axes = (x1_axes,)
    if isinstance(x2_axes, Integral):
        x2_axes = (x2_axes,)
    if isinstance(x1_axes, list):
        x1_axes = tuple(x1_axes)
    if isinstance(x2_axes, list):
        x2_axes = tuple(x2_axes)

    dtype = result_type(x1, x2)

    x1_ind = list(range(x1.ndim))
    x2_ind = list(range(x1.ndim, x1.ndim + x2.ndim))
    out_ind = x1_ind + x2_ind
    adjust_chunks = {}
    for a1, a2 in zip(x1_axes, x2_axes):
        out_ind.remove(x2_ind[a2])
        x2_ind[a2] = x1_ind[a1]
        adjust_chunks[x1_ind[a1]] = lambda c: 1

    out = blockwise(
        _tensordot,
        out_ind,
        x1,
        x1_ind,
        x2,
        x2_ind,
        dtype=dtype,
        adjust_chunks=adjust_chunks,
        axes=(x1_axes, x2_axes),
    )
    return sum(out, axis=x1_axes, dtype=dtype)


def _tensordot(a, b, axes):
    x = np.tensordot(a, b, axes=axes)
    ind = [slice(None, None)] * x.ndim
    for a in sorted(axes[0]):
        ind.insert(a, None)
    x = x[tuple(ind)]
    return x


def vecdot(x1, x2, /, *, axis=-1):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in vecdot")
    return tensordot(x1, x2, axes=((axis,), (axis,)))
