import pytest
from zarr.codecs import BloscCodec, BytesCodec, ZstdCodec
from zarr.storage import LocalStore, ObjectStore

from cubed import config
from cubed.storage.store import open_storage_array
from cubed.storage.zarr import lazy_zarr_array


@pytest.mark.parametrize("use_obstore", [None, False, True])
def test_lazy_zarr_array(tmp_path, use_obstore):
    zarr_path = tmp_path / "lazy.zarr"
    if use_obstore is None:
        kwargs = {}
    else:
        kwargs = dict(storage_options=dict(use_obstore=use_obstore))
    arr = lazy_zarr_array(
        zarr_path,
        shape=(3, 3),
        dtype=int,
        chunks=(2, 2),
        **kwargs,
    )

    assert not zarr_path.exists()
    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        arr.open()

    za = arr.create()
    if use_obstore:
        assert isinstance(za.store, ObjectStore)
    else:
        assert isinstance(za.store, LocalStore)
    assert zarr_path.exists()
    arr.open()


@pytest.mark.parametrize(
    ("compressor", "expected_codecs"),
    [
        (None, (BytesCodec(),)),
        ("auto", (BytesCodec(), ZstdCodec())),
        (
            {"name": "zstd", "configuration": {"level": 1}},
            (BytesCodec(), ZstdCodec(level=1)),
        ),
        (
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 2,
                    "shuffle": "shuffle",
                },
            },
            (
                BytesCodec(),
                BloscCodec(cname="lz4", clevel=2, shuffle="shuffle", typesize=8),
            ),
        ),
    ],
)
def test_compressor(tmp_path, compressor, expected_codecs):
    zarr_path = tmp_path / "lazy.zarr"

    arr = lazy_zarr_array(
        zarr_path, shape=(3, 3), dtype=int, chunks=(2, 2), compressors=compressor
    )
    arr.create()

    # open with zarr python
    with config.set({"storage_name": "zarr-python"}):
        z = open_storage_array(zarr_path, mode="r")

    assert z.metadata.codecs == expected_codecs
