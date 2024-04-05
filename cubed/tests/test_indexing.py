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
        ([6, 7, 2, 9, 10], None),  # add a new dimension
        (None, [6, 7, 2, 9, 10]),  # add a new dimension
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
        (None, slice(None), [2, 1]),
        (slice(None), None, [2, 1]),
        (slice(None), [2, 1], None),
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


def test_multiple_int_array_indexes(spec):
    with pytest.raises(NotImplementedError):
        a = xp.asarray(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            chunks=(2, 2),
            spec=spec,
        )
        a[[1, 2, 1], [2, 1]]
