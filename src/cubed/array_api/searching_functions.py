from cubed.array_api.creation_functions import asarray, zeros_like
from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import _promote_scalars, _real_numeric_dtypes
from cubed.array_api.manipulation_functions import reshape
from cubed.array_api.statistical_functions import max
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import arg_reduction, blockwise, elemwise


def argmax(x, /, *, axis=None, keepdims=False, split_every=None):
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
        split_every=split_every,
    )


def argmin(x, /, *, axis=None, keepdims=False, split_every=None):
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
        split_every=split_every,
    )


def searchsorted(x1, x2, /, *, side="left", sorter=None):
    if x1.ndim != 1:
        raise ValueError("Input array x1 must be one dimensional")

    if sorter is not None:
        raise NotImplementedError(
            "searchsorted with a sorter argument is not supported"
        )

    # call nxp.searchsorted for each pair of blocks in x1 and v
    out = blockwise(
        _searchsorted,
        list(range(x2.ndim + 1)),
        x1,
        [0],
        x2,
        list(range(1, x2.ndim + 1)),
        dtype=nxp.int64,  # TODO: index dtype
        adjust_chunks={0: 1},  # one row for each block in x1
        side=side,
    )

    # add offsets to take account of the position of each block within the array x1
    x1_chunk_sizes = nxp.asarray((0, *x1.chunks[0]))
    x1_chunk_offsets = nxp.cumulative_sum(x1_chunk_sizes)[:-1]
    x1_chunk_offsets = x1_chunk_offsets[(Ellipsis,) + x2.ndim * (nxp.newaxis,)]
    x1_offsets = asarray(x1_chunk_offsets, chunks=1)
    out = where(out < 0, out, out + x1_offsets)

    # combine the results from each block (of a)
    out = max(out, axis=0)

    # fix up any -1 values
    # TODO: use general_blockwise which has block_id to avoid this
    out = where(out >= 0, out, zeros_like(out))

    return out


def _searchsorted(x, y, side):
    res = nxp.searchsorted(x, y, side=side)
    # 0 is only correct for the first block of a, but blockwise doesn't have a way
    # of telling which block is being operated on (unlike map_blocks),
    # so set all 0 values to a special value and set back at the end of searchsorted
    res = nxp.where(res == 0, -1, res)
    return res[nxp.newaxis, :]


def where(condition, x1, x2, /):
    x1, x2 = _promote_scalars(x1, x2, "where")
    dtype = result_type(x1, x2)
    return elemwise(nxp.where, condition, x1, x2, dtype=dtype)
