import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubed.storage.zarr import lazy_empty, lazy_full


def test_lazy_empty(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_empty((3, 3), dtype=int, chunks=(2, 2), store=zarr_path)

    assert not zarr_path.exists()
    with pytest.raises(ValueError):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    arr.open()


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
