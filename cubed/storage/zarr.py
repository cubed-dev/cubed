import hashlib
from typing import Optional, Union

import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


class LazyZarrArray:
    """A Zarr array that may not have been written to storage yet.

    On creation, a normal Zarr array's metadata is immediately written to storage,
    and there is no way to avoid this, hence this class, which separates the creation
    of the object in memory and the creation of array metadata in storage.
    """

    def __init__(
        self,
        store: T_Store,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
        path: Optional[str] = None,
        **kwargs,
    ):
        """Create a Zarr array lazily in memory."""
        # use an empty in-memory Zarr array as a template since it normalizes its properties
        template = zarr.empty(
            shape, dtype=dtype, chunks=chunks, store=zarr.storage.MemoryStore()
        )
        self.shape = template.shape
        self.dtype = template.dtype
        self.chunks = template.chunks
        self.nbytes = template.nbytes

        self.store = store
        self.path = path
        self.kwargs = kwargs

    def create(self, mode: str = "w-") -> zarr.Array:
        """Create the Zarr array in storage.

        The Zarr array's metadata is initialized in the backing store, and any
        initial values are set on the array.

        Parameters
        ----------
        mode : str
            The mode to open the Zarr array with using ``zarr.open``.
            Default is 'w-', which means create, fail it already exists.
        """
        if isinstance(self.store, str) and ("://" in self.store or "::" in self.store):
            store = RandomHashPrefixFSStore(self.store, mode=mode)
        else:
            store = RandomHashPrefixDirectoryStore(self.store)
        target = zarr.open_array(
            store,
            mode=mode,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            path=self.path,
            **self.kwargs,
        )
        return target

    def open(self) -> zarr.Array:
        """Open the Zarr array for reading or writing and return it.

        Note that the Zarr array must have been created or this method will raise an exception.
        """
        # r+ means read/write, fail if it doesn't exist
        if isinstance(self.store, str) and ("://" in self.store or "::" in self.store):
            store = RandomHashPrefixFSStore(self.store, mode="r+")
        else:
            store = RandomHashPrefixDirectoryStore(self.store)
        return zarr.open_array(
            store,
            mode="r+",
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            path=self.path,
        )

    def __repr__(self) -> str:
        return f"cubed.storage.zarr.LazyZarrArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


T_ZarrArray = Union[zarr.Array, LazyZarrArray]


def lazy_zarr_array(
    store: T_Store,
    shape: T_Shape,
    dtype: T_DType,
    chunks: T_RegularChunks,
    path: Optional[str] = None,
    **kwargs,
) -> LazyZarrArray:
    return LazyZarrArray(store, shape, dtype, chunks, path=path, **kwargs)


def open_if_lazy_zarr_array(array: T_ZarrArray) -> zarr.Array:
    """If array is a LazyZarrArray then open it, leaving other arrays unchanged."""
    return array.open() if isinstance(array, LazyZarrArray) else array


class RandomHashPrefixDirectoryStore(zarr.storage.DirectoryStore):
    """Add a random hash prefix to improve load distribution."""

    def __init__(self, path):
        super().__init__(path, normalize_keys=True)

    def _normalize_key(self, key):
        # e.g. 3.0.0.0 -> 78-3.0.0.0
        # note that this only distributes load across one array, not across a workload
        normalized_key = super()._normalize_key(key)
        if normalized_key not in (".zarray", ".zgroup"):  # skip these...
            hash_prefix = hashlib.md5(key.encode("utf-8")).hexdigest()[:2]
            normalized_key = f"{hash_prefix}-{normalized_key}"
        # print(f"Normalizing key {key} to {normalized_key}")
        return normalized_key


class RandomHashPrefixFSStore(zarr.storage.FSStore):
    """Add a random hash prefix to improve load distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, normalize_keys=True, **kwargs)
        print(kwargs)

    def _normalize_key(self, key):
        # e.g. 3.0.0.0 -> 78-3.0.0.0
        # note that this only distributes load across one array, not across a workload
        normalized_key = super()._normalize_key(key)
        if normalized_key not in (".zarray", ".zgroup"):  # skip these...
            hash_prefix = hashlib.md5(key.encode("utf-8")).hexdigest()[:2]
            normalized_key = f"{hash_prefix}-{normalized_key}"
        # print(f"Normalizing key {key} to {normalized_key}")
        return normalized_key
