from pathlib import Path
from typing import Literal, Optional

import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store

# always write empty chunks to avoid a check for each chunk
zarr.config.set({"array.write_empty_chunks": True})


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
    mode: Optional[Literal["r", "r+", "a", "w", "w-"]],
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    **kwargs,
):
    # use obstore if requested
    storage_options = kwargs.get("storage_options", None)
    if storage_options is not None and storage_options.get("use_obstore", False):
        import obstore as obs
        from zarr.storage import ObjectStore

        if isinstance(store, str):
            if "://" not in store:
                p = Path(store)
                store = ObjectStore(obs.store.from_url(p.as_uri(), mkdir=True))
            else:
                store = ObjectStore(obs.store.from_url(store))
        elif isinstance(store, Path):
            p = store
            store = ObjectStore(obs.store.from_url(p.as_uri(), mkdir=True))

    if isinstance(chunks, int):
        chunks = (chunks,)

    if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
        # return zarr.open(
        #     store=store,
        #     mode=mode,
        #     shape=shape,
        #     dtype=dtype,
        #     chunks=chunks,
        #     path=path,
        # )
        group = zarr.open_group(store=store, mode=mode, path=path)
        ret = ZarrV3ArrayGroup(shape=shape, dtype=dtype, chunks=chunks)
        if mode in ("r", "r+"):
            ret["data"] = group["data"]
        else:
            ret["data"] = group.create_array(
                "data",
                shape=shape,
                dtype=dtype,
                chunks=chunks,
            )
        if mode in ("r", "r+"):
            ret["mask"] = group["mask"]
        else:
            ret["mask"] = group.create_array(
                "mask",
                shape=shape,
                dtype=bool,  # TODO: namespace?
                chunks=chunks,
            )
        return ret

    assert mode is not None
    group = zarr.open_group(store=store, mode=mode, path=path)

    # create/open all the arrays in the group
    ret = ZarrV3ArrayGroup(shape=shape, dtype=dtype, chunks=chunks)
    for field in dtype.fields:
        field_dtype, _ = dtype.fields[field]
        if mode in ("r", "r+"):
            ret[field] = group[field]
        else:
            assert chunks is not None
            ret[field] = group.create_array(
                field,
                shape=shape,
                dtype=field_dtype,
                chunks=chunks,
            )
    return ret
