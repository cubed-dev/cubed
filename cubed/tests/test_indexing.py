import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


# Int array indexing is not a part of the array API, so test it separately.


@pytest.mark.parametrize(
    "ind",
    [
        [6, 7, 2, 9, 10],
        ([6, 7, 2, 9, 10], xp.newaxis),
        (xp.newaxis, [6, 7, 2, 9, 10]),
        (slice(None), xp.newaxis),
        (xp.newaxis, slice(None)),
    ],
)
def test_int_array_index_1d(spec, ind):
    a = xp.arange(12, chunks=(3,), spec=spec)
    b = a.rechunk((4,))  # force materialization to test indexing against zarr
    assert_array_equal(b[ind].compute(), np.arange(12)[ind])


@pytest.mark.parametrize(
    "ind",
    [
        (slice(None), [2, 1]),
        ([1, 2, 1], slice(None)),
        (xp.newaxis, slice(None), [2, 1]),
        (slice(None), xp.newaxis, [2, 1]),
        (slice(None), [2, 1], xp.newaxis),
        (xp.newaxis, slice(None), slice(None)),
        (slice(None), xp.newaxis, slice(None)),
        (slice(None), slice(None), xp.newaxis),
    ],
)
def test_int_array_index_2d(spec, ind):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(3, 3),
        spec=spec,
    )
    b = a.rechunk((2, 2))  # force materialization to test indexing against zarr
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert_array_equal(b[ind].compute(), x[ind])


@pytest.mark.parametrize(
    "ind",
    [
        (slice(None), [0, 1, 2, 3]),
        ([0, 1, 2, 3], slice(None)),
    ],
)
def test_int_array_index_2d_no_op(spec, ind):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(3, 3),
        spec=spec,
    )
    assert a is a[ind]


def test_multiple_int_array_indexes(spec):
    with pytest.raises(NotImplementedError):
        a = xp.asarray(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            chunks=(2, 2),
            spec=spec,
        )
        a[[1, 2, 1], [2, 1, 0]]
