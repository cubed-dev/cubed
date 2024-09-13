from typing import Optional, Union

import zarr
from numcodecs.registry import get_codec

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


def open_zarr_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    compressor: Union[dict, str, None] = "default",
    **kwargs,
):
    if isinstance(compressor, dict):
        compressor = get_codec(compressor)

    return zarr.open_array(
        store,
        mode=mode,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        path=path,
        compressor=compressor,
        **kwargs,
    )
