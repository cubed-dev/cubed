import numpy as np
import numpy_groupies as npg
import pytest
from numpy.testing import assert_array_equal

import cubed.array_api as xp
from cubed.backend_array_api import namespace as nxp
from cubed.core.groupby import (
    _get_chunks_for_groups,
    groupby_blockwise,
    groupby_reduction,
)


def test_groupby_reduction_axis0():
    a = xp.full((4 * 6, 5), 7.0, chunks=(4, 2))
    b = xp.asarray([0, 1, 0, 1] * 6, chunks=(4,))
    c = mean_groupby_reduction(a, b, axis=0, num_groups=2)
    assert_array_equal(c.compute(), np.full((2, 5), 7))


def test_groupby_reduction_axis1():
    a = xp.full((5, 4 * 6), 7.0, chunks=(2, 4))
    b = xp.asarray([0, 1, 0, 1] * 6, chunks=(4,))
    c = mean_groupby_reduction(a, b, axis=1, num_groups=2)
    assert_array_equal(c.compute(), np.full((5, 2), 7))


def mean_groupby_reduction(x, by, axis, num_groups):
    intermediate_dtype = [("n", nxp.int64), ("total", nxp.float64)]
    dtype = x.dtype

    return groupby_reduction(
        x,
        by,
        func=_mean_groupby_func,
        combine_func=_mean_groupby_combine,
        aggregate_func=_mean_groupby_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        num_groups=num_groups,
    )


def _mean_groupby_func(a, by, axis, intermediate_dtype, num_groups):
    dtype = dict(intermediate_dtype)
    n = npg.aggregate(by, a, func="len", dtype=dtype["n"], axis=axis, size=num_groups)
    total = npg.aggregate(
        by, a, func="sum", dtype=dtype["total"], axis=axis, size=num_groups
    )
    return {"n": n, "total": total}


def _mean_groupby_combine(a, axis, dummy_axis, dtype, keepdims):
    # only combine over the dummy axis, to preserve grouping along 'axis'
    dtype = dict(dtype)
    n = nxp.sum(a["n"], dtype=dtype["n"], axis=dummy_axis, keepdims=keepdims)
    total = nxp.sum(
        a["total"], dtype=dtype["total"], axis=dummy_axis, keepdims=keepdims
    )
    return {"n": n, "total": total}


def _mean_groupby_aggregate(a, **kwargs):
    return nxp.divide(a["total"], a["n"])


@pytest.mark.parametrize(
    "num_chunks, expected_newchunks, expected_groups_per_chunk",
    [
        [10, (3, 2, 2, 0, 3), 1],
        [5, (3, 2, 2, 0, 3), 1],
        [4, (3, 2, 2, 0, 3), 1],
        [3, (3, 2, 2, 0, 3), 1],
        [2, (5, 2, 3), 2],
        [2, (5, 2, 3), 2],
        [2, (5, 2, 3), 2],
        [2, (5, 2, 3), 2],
        [2, (5, 2, 3), 2],
        [1, (10), 5],
    ],
)
def test_get_chunks_for_groups(
    num_chunks, expected_newchunks, expected_groups_per_chunk
):
    # group 3 has no data
    labels = nxp.asarray([0, 0, 0, 1, 1, 2, 2, 4, 4, 4])
    newchunks, groups_per_chunk = _get_chunks_for_groups(
        num_chunks, labels, num_groups=5
    )
    assert_array_equal(newchunks, expected_newchunks)
    assert groups_per_chunk == expected_groups_per_chunk


def test_groupby_blockwise_axis0():
    a = xp.ones((10, 3), dtype=nxp.int32, chunks=(6, 2))
    b = nxp.asarray([0, 0, 0, 1, 1, 2, 2, 4, 4, 4])
    c = groupby_blockwise(
        a,
        b,
        func=_sum_reduction_func,
        axis=0,
        dtype=nxp.int64,
        num_groups=6,
        groupby_dtype=nxp.int32,
    )
    assert_array_equal(
        c.compute(),
        nxp.asarray(
            [
                [3, 3, 3],
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],  # group 3 has no data
                [3, 3, 3],
                [0, 0, 0],  # final group since we specified num_groups=6
            ]
        ),
    )


def test_groupby_blockwise_axis1():
    a = xp.ones((3, 10), dtype=nxp.int32, chunks=(6, 2))
    b = nxp.asarray([0, 0, 0, 1, 1, 2, 2, 4, 4, 4])
    c = groupby_blockwise(
        a,
        b,
        func=_sum_reduction_func,
        axis=1,
        dtype=nxp.int64,
        num_groups=6,
        groupby_dtype=nxp.int32,
    )
    assert_array_equal(
        c.compute(),
        nxp.asarray(
            [
                [3, 2, 2, 0, 3, 0],
                [3, 2, 2, 0, 3, 0],
                [3, 2, 2, 0, 3, 0],
            ]
        ),
    )


def _sum_reduction_func(arr, by, axis, start_group, num_groups, groupby_dtype):
    # change 'by' so it starts from 0 for each chunk
    by = by - start_group
    return npg.aggregate(
        by, arr, func="sum", dtype=groupby_dtype, axis=axis, size=num_groups
    )
