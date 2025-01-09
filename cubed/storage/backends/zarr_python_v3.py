from typing import Optional

import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


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

    if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
        return zarr.open(
            store=store,
            mode=mode,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            path=path,
        )

    group = zarr.open_group(store=store, mode=mode, path=path)

    # create/open all the arrays in the group
    ret = ZarrV3ArrayGroup(shape=shape, dtype=dtype, chunks=chunks)
    for field in dtype.fields:
        field_dtype, _ = dtype.fields[field]
        if mode in ("r", "r+"):
            ret[field] = group[field]
        else:
            ret[field] = group.create_array(
                field,
                shape=shape,
                dtype=field_dtype,
                chunks=chunks,
            )
    return ret
