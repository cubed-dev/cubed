import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubed.storage.zarr import lazy_empty, lazy_from_array, lazy_full


def test_lazy_empty(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_empty((3, 3), dtype=int, chunks=(2, 2), store=zarr_path)

    assert not zarr_path.exists()
    with pytest.raises(ValueError):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    arr.open()


def test_lazy_from_array(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
    arr = lazy_from_array(a, dtype=a.dtype, chunks=(2, 2), store=zarr_path)

    assert not zarr_path.exists()
    with pytest.raises(ValueError):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    assert_array_equal(arr.open()[:], a)


def test_lazy_full(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_full(
        (3, 3),
        1,
        dtype=int,
        chunks=(2, 2),
        store=zarr_path,
    )

    assert not zarr_path.exists()
    with pytest.raises(ValueError):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    assert_array_equal(arr.open()[:], np.full((3, 3), 1, dtype=int))
