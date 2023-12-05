from numbers import Integral
from typing import Any

import numpy as np
import zarr
from zarr.indexing import BasicIndexer, is_slice

from cubed.backend_array_api import namespace as nxp
from cubed.backend_array_api import numpy_array_to_backend_array
from cubed.types import T_DType, T_RegularChunks, T_Shape


class VirtualEmptyArray:
    """An array that is never materialized (in memory or on disk) and contains empty values."""

    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
    ):
        # use an empty in-memory Zarr array as a template since it normalizes its properties
        template = zarr.empty(
            shape, dtype=dtype, chunks=chunks, store=zarr.storage.MemoryStore()
        )
        self.shape = template.shape
        self.dtype = template.dtype
        self.chunks = template.chunks
        self.template = template

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        indexer = BasicIndexer(key, self.template)
        return nxp.empty(indexer.shape, dtype=self.dtype)

    @property
    def oindex(self):
        return self.template.oindex


class VirtualFullArray:
    """An array that is never materialized (in memory or on disk) and contains a single fill value."""

    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
        fill_value: Any = None,
    ):
        # use an empty in-memory Zarr array as a template since it normalizes its properties
        template = zarr.full(
            shape,
            fill_value,
            dtype=dtype,
            chunks=chunks,
            store=zarr.storage.MemoryStore(),
        )
        self.shape = template.shape
        self.dtype = template.dtype
        self.chunks = template.chunks
        self.template = template
        self.fill_value = fill_value

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        indexer = BasicIndexer(key, self.template)
        return nxp.full(indexer.shape, fill_value=self.fill_value, dtype=self.dtype)

    @property
    def oindex(self):
        return self.template.oindex


class VirtualOffsetsArray:
    """An array that is never materialized (in memory or on disk) and contains sequentially incrementing integers."""

    def __init__(self, shape: T_Shape):
        dtype = np.int32
        chunks = (1,) * len(shape)
        # use an empty in-memory Zarr array as a template since it normalizes its properties
        template = zarr.empty(
            shape, dtype=dtype, chunks=chunks, store=zarr.storage.MemoryStore()
        )
        self.shape = template.shape
        self.dtype = template.dtype
        self.chunks = template.chunks
        self.ndim = template.ndim

    def __getitem__(self, key):
        if key == () and self.shape == ():
            return nxp.asarray(0, dtype=self.dtype)
        return numpy_array_to_backend_array(
            np.ravel_multi_index(_key_to_index_tuple(key), self.shape), dtype=self.dtype
        )


class VirtualInMemoryArray:
    """A small array that is held in memory but never materialized on disk."""

    def __init__(
        self,
        array: np.ndarray,  # TODO: generalise
        chunks: T_RegularChunks,
    ):
        self.array = array
        # use an in-memory Zarr array as a template since it normalizes its properties
        # and is needed for oindex
        template = zarr.empty(
            array.shape,
            dtype=array.dtype,
            chunks=chunks,
            store=zarr.storage.MemoryStore(),
        )
        self.shape = template.shape
        self.dtype = template.dtype
        self.chunks = template.chunks
        self.template = template
        if array.size > 0:
            template[...] = array

    def __getitem__(self, key):
        return self.array.__getitem__(key)

    @property
    def oindex(self):
        return self.template.oindex


def _key_to_index_tuple(selection):
    if isinstance(selection, slice):
        selection = (selection,)
    assert all(isinstance(s, (slice, Integral)) for s in selection)
    sel = []
    for s in selection:
        if isinstance(s, Integral):
            sel.append(s)
        elif is_slice(s) and s.stop == s.start + 1 and (s.step is None or s.step == 1):
            sel.append(s.start)
        else:
            raise NotImplementedError(f"Offset selection not supported: {selection}")
    return tuple(sel)


def virtual_empty(
    shape: T_Shape, *, dtype: T_DType, chunks: T_RegularChunks, **kwargs
) -> VirtualEmptyArray:
    return VirtualEmptyArray(shape, dtype, chunks, **kwargs)


def virtual_full(
    shape: T_Shape,
    fill_value: Any,
    *,
    dtype: T_DType,
    chunks: T_RegularChunks,
    **kwargs,
) -> VirtualFullArray:
    return VirtualFullArray(shape, dtype, chunks, fill_value, **kwargs)


def virtual_offsets(shape: T_Shape) -> VirtualOffsetsArray:
    return VirtualOffsetsArray(shape)


def virtual_in_memory(
    array: np.ndarray,
    chunks: T_RegularChunks,
) -> VirtualInMemoryArray:
    return VirtualInMemoryArray(array, chunks)
