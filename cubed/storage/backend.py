from typing import Optional

from cubed import config
from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


def open_backend_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    **kwargs,
):
    # get storage name from top-level config
    # e.g. set globally with CUBED_STORAGE_NAME=tensorstore
    storage_name = config.get("storage_name", None)

    if storage_name is None or storage_name == "zarr-python":
        from cubed.storage.backends.zarr_python import open_zarr_array

        open_func = open_zarr_array
    else:
        raise ValueError(f"Unrecognized storage name: {storage_name}")

    return open_func(
        store,
        mode,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        path=path,
        **kwargs,
    )
