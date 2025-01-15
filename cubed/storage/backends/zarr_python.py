# type: ignore  # Zarr Python v2 types are different to v3
from typing import Optional, Union

import zarr
from numcodecs.registry import get_codec

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store
from cubed.utils import join_path


class ZarrArrayGroup(dict):
    def __init__(
        self,
        shape: Optional[T_Shape] = None,
        dtype: Optional[T_DType] = None,
        chunks: Optional[T_RegularChunks] = None,
    ):
        dict.__init__(self)
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        return {field: zarray[key] for field, zarray in self.items()}

    def set_basic_selection(self, selection, value, fields=None):
        self[fields][selection] = value


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

    if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
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
    else:
        ret = ZarrArrayGroup(shape=shape, dtype=dtype, chunks=chunks)
        for field in dtype.fields:
            field_dtype, _ = dtype.fields[field]
            field_path = field if path is None else join_path(path, field)
            ret[field] = zarr.open_array(
                store,
                mode=mode,
                shape=shape,
                dtype=field_dtype,
                chunks=chunks,
                path=field_path,
                compressor=compressor,
                **kwargs,
            )
        return ret
