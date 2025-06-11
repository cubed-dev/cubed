from typing import Optional

from cubed import config
from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


def backend_storage_name():
    # get storage name from top-level config
    # e.g. set globally with CUBED_STORAGE_NAME=tensorstore
    storage_name = config.get("storage_name", None)

    if storage_name is None:
        import zarr

        if zarr.__version__[0] == "3":
            storage_name = "zarr-python-v3"
        else:
            storage_name = "zarr-python"

    return storage_name


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
    storage_name = backend_storage_name()

    if storage_name == "zarr-python":
        from cubed.storage.backends.zarr_python import open_zarr_array  # type: ignore

        open_func = open_zarr_array

        # set object codec if needed
        import numpy as np

        if np.dtype(dtype).hasobject and "object_codec" not in kwargs:
            import numcodecs

            object_codec = numcodecs.Pickle()
            kwargs["object_codec"] = object_codec

    elif storage_name == "zarr-python-v3":
        from cubed.storage.backends.zarr_python_v3 import open_zarr_v3_array

        open_func = open_zarr_v3_array
    elif storage_name == "tensorstore":
        from cubed.storage.backends.tensorstore import open_tensorstore_array

        open_func = open_tensorstore_array
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
