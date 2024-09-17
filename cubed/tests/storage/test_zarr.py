import pytest
import zarr
from numcodecs.registry import get_codec

from cubed import config
from cubed.storage.backend import open_backend_array
from cubed.storage.zarr import lazy_zarr_array

ZARR_PYTHON_V3 = zarr.__version__[0] == "3"


def test_lazy_zarr_array(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_zarr_array(zarr_path, shape=(3, 3), dtype=int, chunks=(2, 2))

    assert not zarr_path.exists()
    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    arr.open()


@pytest.mark.skipif(
    ZARR_PYTHON_V3, reason="setting zarr compressor not yet possible for Zarr Python v3"
)
@pytest.mark.parametrize(
    "compressor",
    [
        None,
        {"id": "zstd", "level": 1},
        {"id": "blosc", "cname": "lz4", "clevel": 2, "shuffle": -1},
    ],
)
def test_compression(tmp_path, compressor):
    zarr_path = tmp_path / "lazy.zarr"

    arr = lazy_zarr_array(
        zarr_path, shape=(3, 3), dtype=int, chunks=(2, 2), compressor=compressor
    )
    arr.create()

    # open with zarr python (for zarr python v2 and tensorstore)
    with config.set({"storage_name": "zarr-python"}):
        z = open_backend_array(zarr_path, mode="r")

    if compressor is None:
        assert z.compressor is None
    else:
        assert z.compressor == get_codec(compressor)
