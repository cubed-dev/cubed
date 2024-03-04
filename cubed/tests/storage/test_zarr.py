import pytest

from cubed.storage.zarr import lazy_zarr_array


def test_lazy_zarr_array(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_zarr_array(zarr_path, shape=(3, 3), dtype=int, chunks=(2, 2))

    assert not zarr_path.exists()
    with pytest.raises(ValueError):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    arr.open()
