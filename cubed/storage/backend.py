from typing import Optional

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


def open_backend_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    storage_name: Optional[str] = None,
    **kwargs,
):
    if storage_name is None:
        storage_name = "zarr"

    if storage_name == "tensorstore":
        from cubed.storage.backends.tensorstore import open_tensorstore_array

        open_func = open_tensorstore_array
    elif storage_name == "zarr":
        from cubed.storage.backends.zarr import open_zarr_array

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
