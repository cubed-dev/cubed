from operator import mul
from typing import Optional, Union

import numpy as np
import zarr
from toolz import reduce

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
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.chunks = chunks
        self.store = store
        self.path = path
        self.kwargs = kwargs

    @property
    def size(self):
        """Number of elements in the array."""
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self) -> int:
        """Number of bytes in array"""
        return self.size * self.dtype.itemsize

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
            path=self.path,
            **self.kwargs,
        )
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
            path=self.path,
            **self.kwargs,
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
    return LazyZarrArray(
        store,
        shape,
        dtype,
        chunks,
        path=path,
        **kwargs,
    )


def open_if_lazy_zarr_array(array: T_ZarrArray) -> zarr.Array:
    """If array is a LazyZarrArray then open it, leaving other arrays unchanged."""
    return array.open() if isinstance(array, LazyZarrArray) else array
