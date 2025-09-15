import numpy as np
import pytest

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal


@pytest.fixture
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
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    with pytest.raises(NotImplementedError):
        a[[1, 2, 1], [2, 1, 0]]


def test_blocks():
    # based on dask tests
    x = xp.arange(10, chunks=2)
    assert x.blocks.shape == (5,)
    assert x.blocks.size == 5

    assert_array_equal(x.blocks[0], x[:2])
    assert_array_equal(x.blocks[-1], x[-2:])
    assert_array_equal(x.blocks[:3], x[:6])
    assert_array_equal(x.blocks[[0, 1, 2]], x[:6])
    assert_array_equal(x.blocks[[3, 0, 2]], np.array([6, 7, 0, 1, 4, 5]))

    x = cubed.random.random((20, 20), chunks=(4, 5))
    assert x.blocks.shape == (5, 4)
    assert x.blocks.size == 20
    assert_array_equal(x.blocks[0], x[:4])
    assert_array_equal(x.blocks[0, :3], x[:4, :15])
    assert_array_equal(x.blocks[:, :3], x[:, :15])

    x = xp.ones((40, 40, 40), chunks=(10, 10, 10))
    assert_array_equal(x.blocks[0, :, 0], np.ones((10, 40, 10)))

    x = xp.ones((2, 2), chunks=1)
    with pytest.raises(ValueError, match="newaxis is not supported"):
        x.blocks[xp.newaxis, :, :]
    with pytest.raises(NotImplementedError):
        x.blocks[[0, 1], [0, 1]]
    with pytest.raises(IndexError, match="out of bounds"):
        x.blocks[100, 100]
