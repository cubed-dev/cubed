from numbers import Integral
from typing import Any

import numpy as np
from ndindex import ndindex

from cubed.backend_array_api import namespace as nxp
from cubed.backend_array_api import numpy_array_to_backend_array
from cubed.storage.types import ArrayMetadata
from cubed.types import T_DType, T_RegularChunks, T_Shape
from cubed.utils import array_memory, broadcast_trick, memory_repr


class VirtualEmptyArray(ArrayMetadata):
    """An array that is never materialized (in memory or on disk) and contains empty values."""

    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
    ):
        super().__init__(shape, dtype, chunks)

    def __getitem__(self, key):
        idx = ndindex[key]
        newshape = idx.newshape(self.shape)
        # use broadcast trick so array chunks only occupy a single value in memory
        return broadcast_trick(nxp.empty)(newshape, dtype=self.dtype)

    @property
    def chunkmem(self):
        # take broadcast trick into account
        return array_memory(self.dtype, (1,))


class VirtualFullArray(ArrayMetadata):
    """An array that is never materialized (in memory or on disk) and contains a single fill value."""

    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
        fill_value: Any = None,
    ):
        super().__init__(shape, dtype, chunks)
        self.fill_value = fill_value

    def __getitem__(self, key):
        idx = ndindex[key]
        newshape = idx.newshape(self.shape)
        # use broadcast trick so array chunks only occupy a single value in memory
        return broadcast_trick(nxp.full)(
            newshape, fill_value=self.fill_value, dtype=self.dtype
        )

    @property
    def chunkmem(self):
        # take broadcast trick into account
        return array_memory(self.dtype, (1,))


class VirtualOffsetsArray(ArrayMetadata):
    """An array that is never materialized (in memory or on disk) and contains sequentially incrementing integers."""

    def __init__(self, shape: T_Shape):
        dtype = nxp.int32
        chunks = (1,) * len(shape)
        super().__init__(shape, dtype, chunks)

    def __getitem__(self, key):
        if key == () and self.shape == ():
            return nxp.asarray(0, dtype=self.dtype)
        return numpy_array_to_backend_array(
            np.ravel_multi_index(_key_to_index_tuple(key), self.shape), dtype=self.dtype
        )


class VirtualInMemoryArray(ArrayMetadata):
    """A small array that is held in memory but never materialized on disk."""

    def __init__(
        self,
        array: np.ndarray,  # TODO: generalise to array API type
        chunks: T_RegularChunks,
        max_nbytes: int = 10**6,
    ):
        if array.nbytes > max_nbytes:
            raise ValueError(
                f"Size of in memory array is {memory_repr(array.nbytes)} which exceeds maximum of {memory_repr(max_nbytes)}. Consider loading the array from storage using `from_array`."
            )
        self.array = array
        super().__init__(array.shape, array.dtype, chunks)

    def __getitem__(self, key):
        return self.array.__getitem__(key)


def _key_to_index_tuple(selection):
    if isinstance(selection, slice):
        selection = (selection,)
    assert all(isinstance(s, (slice, Integral)) for s in selection)
    sel = []
    for s in selection:
        if isinstance(s, Integral):
            sel.append(s)
        elif (
            isinstance(s, slice)
            and s.stop == s.start + 1
            and (s.step is None or s.step == 1)
        ):
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
