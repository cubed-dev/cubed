import numpy as np

from cubed.array_api.data_type_functions import result_type
from cubed.core import blockwise, reduction, squeeze


def matmul(x1, x2, /):
    assert x1.ndim >= 2
    assert x2.ndim >= 2
    assert x1.ndim == x2.ndim

    out_ind = tuple(range(x1.ndim + 1))
    lhs_ind = tuple(range(x1.ndim))
    rhs_ind = tuple(range(x1.ndim - 2)) + (lhs_ind[-1], x1.ndim)

    dtype = result_type(x1, x2)

    out = blockwise(
        _matmul,
        out_ind,
        x1,
        lhs_ind,
        x2,
        rhs_ind,
        adjust_chunks={lhs_ind[-1]: 1},
        dtype=dtype,
    )

    out = _sum_wo_cat(out, axis=-2, dtype=dtype)

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


def outer(x1, x2, /):
    return blockwise(np.outer, "ij", x1, "i", x2, "j", dtype=x1.dtype)
