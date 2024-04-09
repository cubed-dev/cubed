import numpy as np
import numpy_groupies as npg
from numpy.testing import assert_array_equal

import cubed.array_api as xp
from cubed.backend_array_api import namespace as nxp
from cubed.core.groupby import groupby_reduction


def test_groupby_reduction_axis0():
    a = xp.full((4 * 6, 5), 7, dtype=nxp.int32, chunks=(4, 2))
    b = xp.asarray([0, 1, 0, 1] * 6, chunks=(4,))
    c = mean_groupby_reduction(a, b, axis=0, num_groups=2)
    assert_array_equal(c.compute(), np.full((2, 5), 7))


def test_groupby_reduction_axis1():
    a = xp.full((5, 4 * 6), 7, dtype=nxp.int32, chunks=(2, 4))
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
        aggegrate_func=_mean_groupby_aggregate,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        num_groups=num_groups,
    )


def _mean_groupby_func(a, by, axis, intermediate_dtype, num_groups):
    dtype = dict(intermediate_dtype)  # TODO: shouldn't have to do this
    n = npg.aggregate(by, a, func="len", dtype=dtype["n"], axis=axis, size=num_groups)
    total = npg.aggregate(
        by, a, func="sum", dtype=dtype["total"], axis=axis, size=num_groups
    )
    return {"n": n, "total": total}


def _mean_groupby_combine(a, axis, dummy_axis, dtype, keepdims):
    # only combine over the dummy axis, to preserve grouping along 'axis'
    dtype = dict(dtype)  # TODO: shouldn't have to do this
    n = nxp.sum(a["n"], dtype=dtype["n"], axis=dummy_axis, keepdims=keepdims)
    total = nxp.sum(
        a["total"], dtype=dtype["total"], axis=dummy_axis, keepdims=keepdims
    )
    return {"n": n, "total": total}


def _mean_groupby_aggregate(a):
    return nxp.divide(a["total"], a["n"])
