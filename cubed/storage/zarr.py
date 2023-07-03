from typing import Any, Optional, Union

import zarr
from numpy import ndarray

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


class LazyZarrArray:
    """A Zarr array that may not have been written to storage yet.

    On creation, a normal Zarr array's metadata is immediately written to storage,
    and there is no way to avoid this, hence this class, which separates the creation
    of the object in memory and the creation of array metadata in storage.
    """

    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
        store: T_Store,
        initial_values: Optional[ndarray] = None,
        fill_value: Any = None,
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

        self.store = store
        self.initial_values = initial_values
        self.fill_value = fill_value
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
        target = zarr.open_array(
            self.store,
            mode=mode,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            fill_value=self.fill_value,
            **self.kwargs,
        )
        if self.initial_values is not None and self.initial_values.size > 0:
            target[...] = self.initial_values
        return target

    def open(self) -> zarr.Array:
        """Open the Zarr array for reading or writing and return it.

        Note that the Zarr array must have been created or this method will raise an exception.
        """
        # r+ means read/write, fail if it doesn't exist
        return zarr.open_array(
            self.store,
            mode="r+",
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
        )

    def __repr__(self) -> str:
        return f"cubed.storage.zarr.LazyZarrArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


T_ZarrArray = Union[zarr.Array, LazyZarrArray]


def lazy_empty(
    shape: T_Shape, *, dtype: T_DType, chunks: T_RegularChunks, store: T_Store, **kwargs
) -> LazyZarrArray:
    return LazyZarrArray(shape, dtype, chunks, store, **kwargs)


def lazy_from_array(
    array: ndarray, *, dtype: T_DType, chunks: T_RegularChunks, store: T_Store, **kwargs
) -> LazyZarrArray:
    return LazyZarrArray(
        array.shape, dtype, chunks, store, initial_values=array, **kwargs
    )


def lazy_full(
    shape: T_Shape,
    fill_value: Any,
    *,
    dtype: T_DType,
    chunks: T_RegularChunks,
    store: T_Store,
    **kwargs,
) -> LazyZarrArray:
    return LazyZarrArray(shape, dtype, chunks, store, fill_value=fill_value, **kwargs)


def open_if_lazy_zarr_array(array: T_ZarrArray) -> zarr.Array:
    """If array is a LazyZarrArray then open it, leaving other arrays unchanged."""
    return array.open() if isinstance(array, LazyZarrArray) else array
