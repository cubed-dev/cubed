import pytest

from cubed.storage.zarr import lazy_empty


def test_lazy_empty(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_empty((3, 3), dtype=int, chunks=(2, 2), store=zarr_path)

    assert not zarr_path.exists()
    with pytest.raises(ValueError):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    arr.open()
