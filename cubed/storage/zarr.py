import dataclasses
import math
from typing import Any, Union

import numpy as np
import tensorstore
import zarr

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


@dataclasses.dataclass(frozen=True)
class TensorstoreArray:
    array: tensorstore.TensorStore

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype.numpy_dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self.array.chunk_layout.read_chunk.shape or ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def oindex(self):
        return self.array.oindex

    def __getitem__(self, key):
        # read eagerly
        return self.array.__getitem__(key).read().result()

    def __setitem__(self, key, value):
        self.array.__setitem__(key, value)


def encode_dtype(d):
    if d.fields is None:
        return d.str
    else:
        return d.descr


def open_tensorstore_array(
    store: T_Store,
    mode: str,
    *,
    shape: T_Shape,
    dtype: T_DType,
    chunks: T_RegularChunks,
    fill_value: Any = None,
    **kwargs,
):
    store = str(store)  # TODO: check if Path or str
    if mode == "r":
        open_kwargs = dict(read=True, open=True)
    if mode == "r+":
        open_kwargs = dict(read=True, write=True, open=True)
    elif mode == "a":
        open_kwargs = dict(read=True, write=True, open=True, create=True)
    elif mode == "w":
        open_kwargs = dict(write=True, create=True, delete_existing=True)
    elif mode == "w-":
        open_kwargs = dict(write=True, create=True)
    else:
        raise ValueError(f"Mode not supported: {mode}")
    if dtype.fields is None:
        return TensorstoreArray(
            tensorstore.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": store},
                    "metadata": {
                        "chunks": chunks,
                        "dtype": encode_dtype(dtype),
                    },
                },
                shape=shape,
                dtype=dtype,
                fill_value=fill_value,
                **open_kwargs,
            ).result()
        )
    else:
        ret = {}
        for field in dtype.fields:
            field_dtype, _ = dtype.fields[field]
            target = TensorstoreArray(
                tensorstore.open(
                    {
                        "driver": "zarr",
                        "kvstore": {"driver": "file", "path": store},
                        "field": field,
                        "metadata": {
                            "chunks": chunks,
                            "dtype": encode_dtype(dtype),
                        },
                    },
                    shape=shape,
                    dtype=field_dtype,
                    fill_value=fill_value,
                    **open_kwargs,
                ).result()
            )
            ret[field] = target
        return ret


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
        target = open_tensorstore_array(
            self.store,
            mode=mode,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            fill_value=self.fill_value,
            **self.kwargs,
        )
        return target

    def open(self) -> zarr.Array:
        """Open the Zarr array for reading or writing and return it.

        Note that the Zarr array must have been created or this method will raise an exception.
        """
        # r+ means read/write, fail if it doesn't exist
        return open_tensorstore_array(
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
