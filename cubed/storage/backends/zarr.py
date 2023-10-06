from typing import Optional

import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


def open_zarr_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    **kwargs,
):
    return zarr.open_array(
        store,
        mode=mode,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        path=path,
        **kwargs,
    )
