import pytest
import zarr
from marray import numpy as mxp

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal


@pytest.fixture
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


def test_marray_elemwise(spec):
    a = xp.arange(4, spec=spec)
    b = a + 1
    res = b.compute()
    assert_array_equal(res.data, [1, 2, 3, 4])
    assert_array_equal(res.mask, [False, False, False, False])


def test_marray_max(spec):
    ma = mxp.asarray([1, 2, 3, 4], mask=[False, True, False, True])
    a = xp.asarray(ma, chunks=2, spec=spec)
    b = xp.max(a)
    res = b.compute()
    assert_array_equal(res.data, [3])
    assert_array_equal(res.mask, [False])


def test_marray_mean(spec):
    ma = mxp.asarray([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, True])
    a = xp.asarray(ma, chunks=2, spec=spec)
    b = xp.mean(a)
    res = b.compute()
    assert_array_equal(res.data, [2.0])
    assert_array_equal(res.mask, [False])


def test_marray_filtering_using_masks(tmp_path, spec):
    # create a regular zarr array (no masks)
    store = store = tmp_path / "source.zarr"
    za = zarr.create_array(store=store, shape=(4,), chunks=(2,), dtype="int32")
    za[:] = [1, 2, 3, 4]

    a = cubed.from_array(za, spec=spec)
    b = a % 2 == 0  # filter out even values

    # create a cubed masked array from a and b
    c = cubed.map_blocks(
        lambda x, y: mxp.asarray(x.data, mask=y.data), a, b, dtype=a.dtype
    )

    res = c.compute()
    assert res.data[0] == 1
    assert res.data[2] == 3
    assert_array_equal(res.mask, [False, True, False, True])
