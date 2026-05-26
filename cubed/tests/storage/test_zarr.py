import pytest
from zarr.codecs import BloscCodec, BytesCodec, ZstdCodec

from cubed import config
from cubed.storage.store import open_storage_array
from cubed.storage.zarr import lazy_zarr_array


def test_lazy_zarr_array(tmp_path):
    zarr_path = tmp_path / "lazy.zarr"
    arr = lazy_zarr_array(zarr_path, shape=(3, 3), dtype=int, chunks=(2, 2))

    assert not zarr_path.exists()
    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        arr.open()

    arr.create()
    assert zarr_path.exists()
    arr.open()


@pytest.mark.parametrize(
    ("codecs", "expected_codecs"),
    [
        (None, (BytesCodec(), ZstdCodec())),
        (({"name": "bytes"},), (BytesCodec(),)),
        (
            ({"name": "bytes"}, {"name": "zstd", "configuration": {"level": 1}}),
            (BytesCodec(), ZstdCodec(level=1)),
        ),
        (
            (
                {"name": "bytes"},
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "lz4",
                        "clevel": 2,
                        "shuffle": "shuffle",
                    },
                },
            ),
            (
                BytesCodec(),
                BloscCodec(cname="lz4", clevel=2, shuffle="shuffle", typesize=8),
            ),
        ),
    ],
)
def test_zarr_codecs(tmp_path, codecs, expected_codecs):
    zarr_path = tmp_path / "lazy.zarr"

    arr = lazy_zarr_array(
        zarr_path, shape=(3, 3), dtype=int, chunks=(2, 2), codecs=codecs
    )
    arr.create()

    # open with zarr python
    with config.set({"storage_name": "zarr-python"}):
        z = open_storage_array(zarr_path, mode="r")

    assert z.metadata.codecs == expected_codecs
