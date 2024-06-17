from typing import Optional

import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store
from cubed.utils import join_path


class ZarrV3ArrayGroup(dict):
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


def open_zarr_v3_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    **kwargs,
):
    if isinstance(chunks, int):
        chunks = (chunks,)

    if mode in ("r", "r+"):
        # TODO: remove when https://github.com/zarr-developers/zarr-python/issues/1978 is fixed
        if mode == "r+":
            mode = "w"
        if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
            return zarr.open(store=store, mode=mode, path=path)
        else:
            ret = ZarrV3ArrayGroup(shape=shape, dtype=dtype, chunks=chunks)
            for field in dtype.fields:
                field_dtype, _ = dtype.fields[field]
                field_path = field if path is None else join_path(path, field)
                ret[field] = zarr.open(store=store, mode=mode, path=field_path)
            return ret
    else:
        overwrite = True if mode == "a" else False
        if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
            return zarr.create(
                shape=shape,
                dtype=dtype,
                chunk_shape=chunks,
                store=store,
                overwrite=overwrite,
                path=path,
            )
        else:
            ret = ZarrV3ArrayGroup(shape=shape, dtype=dtype, chunks=chunks)
            for field in dtype.fields:
                field_dtype, _ = dtype.fields[field]
                field_path = field if path is None else join_path(path, field)
                ret[field] = zarr.create(
                    shape=shape,
                    dtype=field_dtype,
                    chunk_shape=chunks,
                    store=store,
                    overwrite=overwrite,
                    path=field_path,
                )
            return ret
