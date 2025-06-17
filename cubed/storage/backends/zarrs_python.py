import zarr
import zarrs  # noqa: F401

# re-export the Zarr v3 open function
from cubed.storage.backends.zarr_python_v3 import open_zarr_v3_array  # noqa: F401

# need to set zarr config after importing from zarr_python_v3 to ensure pipeline is set
zarr.config.set(
    {
        "array.write_empty_chunks": True,
        "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    }
)
