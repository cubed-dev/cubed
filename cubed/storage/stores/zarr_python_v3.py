from pathlib import Path
from typing import Literal, Optional

import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store
from cubed.utils import is_cloud_storage_path

try:
    import obstore
except ImportError:
    obstore = None  # type: ignore

zarr.config.set(
    {
        # always write empty chunks to avoid a check for each chunk
        "array.write_empty_chunks": True,
        # enable rectilinear (irregular) chunk grids, needed for allow_irregular rechunking
        "array.rectilinear_chunks": True,
    }
)


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
    # use obstore if explicitly requested, or if library is installed and store is a cloud store
    storage_options = kwargs.pop("storage_options", None)
    obstore_requested = storage_options is not None and storage_options.get(
        "use_obstore", False
    )
    obstore_installed = obstore is not None
    if obstore_requested and not obstore_installed:
        raise RuntimeError(
            "obstore was requested with 'use_obstore=True' but it is not installed"
        )
    use_obstore = obstore_requested or (
        obstore_installed
        and isinstance(store, (str, Path))
        and is_cloud_storage_path(store)
    )
    if use_obstore:
        from zarr.storage import ObjectStore

        if isinstance(store, str):
            if "://" not in store:
                p = Path(store)
                store = ObjectStore(obstore.store.from_url(p.as_uri(), mkdir=True))
            else:
                store = ObjectStore(obstore.store.from_url(store))
        elif isinstance(store, Path):
            p = store
            store = ObjectStore(obstore.store.from_url(p.as_uri(), mkdir=True))
        else:
            raise RuntimeError("Store must be a string or `Path` object for obstore")

    if isinstance(chunks, int):
        chunks = (chunks,)

    if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
        if mode in ("r", "r+"):
            # ignore type warning since Zarr can handle path=None
            return zarr.open_array(store=store, path=path)  # type: ignore[arg-type]

        try:
            return zarr.create_array(
                store=store,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                name=path,
                **kwargs,
            )
        except zarr.errors.ContainsArrayError as e:
            if mode == "a":
                return zarr.open_array(store=store, path=path)  # type: ignore[arg-type]
            raise e

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
                **kwargs,
            )
    return ret
