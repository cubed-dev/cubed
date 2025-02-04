from numbers import Integral
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

from cubed.backend_array_api import namespace as nxp
from cubed.backend_array_api import numpy_array_to_backend_array
from cubed.storage.types import ArrayMetadata
from cubed.types import T_DType, T_RegularChunks, T_Shape, T_StandardArray
from cubed.utils import array_memory, broadcast_trick, memory_repr

if TYPE_CHECKING:
    from zarr.core.indexing import Selection


class VirtualArray(ArrayMetadata):
    pass


class VirtualEmptyArray(VirtualArray):
    """An array that is never materialized (in memory or on disk) and contains empty values."""

    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
    ):
        super().__init__(shape, dtype, chunks)

    def __getitem__(self, key: "Selection") -> T_StandardArray:
        from ndindex import ndindex  # import as needed to avoid slow 'import cubed'

        idx = ndindex[key]
        newshape = idx.newshape(self.shape)
        # use broadcast trick so array chunks only occupy a single value in memory
        return broadcast_trick(nxp.empty)(newshape, dtype=self.dtype)

    @property
    def chunkmem(self) -> int:
        # take broadcast trick into account
        return array_memory(self.dtype, (1,))


class VirtualFullArray(VirtualArray):
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

    def __getitem__(self, key: "Selection") -> T_StandardArray:
        from ndindex import ndindex  # import as needed to avoid slow 'import cubed'

        idx = ndindex[key]
        newshape = idx.newshape(self.shape)
        # use broadcast trick so array chunks only occupy a single value in memory
        return broadcast_trick(nxp.full)(
            newshape, fill_value=self.fill_value, dtype=self.dtype
        )

    @property
    def chunkmem(self) -> int:
        # take broadcast trick into account
        return array_memory(self.dtype, (1,))


class VirtualOffsetsArray(VirtualArray):
    """An array that is never materialized (in memory or on disk) and contains sequentially incrementing integers."""

    def __init__(self, shape: T_Shape):
        dtype = nxp.int32
        chunks = (1,) * len(shape)
        super().__init__(shape, dtype, chunks)

    def __getitem__(self, key: "Selection") -> T_StandardArray:
        if key == () and self.shape == ():
            return nxp.asarray(0, dtype=self.dtype)
        return numpy_array_to_backend_array(
            np.ravel_multi_index(_key_to_index_tuple(key), self.shape),
            dtype=self.dtype,  # type: ignore[arg-type]
        )


class VirtualInMemoryArray(VirtualArray):
    """A small array that is held in memory but never materialized on disk."""

    def __init__(
        self,
        array: T_StandardArray,
        chunks: T_RegularChunks,
        max_nbytes: int = 10**6,
    ):
        nbytes = array_memory(array.dtype, array.shape)
        if nbytes > max_nbytes:
            raise ValueError(
                f"Size of in memory array is {memory_repr(nbytes)} which exceeds maximum of {memory_repr(max_nbytes)}. Consider loading the array from storage using `from_array`."
            )
        self.array = array
        super().__init__(array.shape, array.dtype, chunks)

    def __getitem__(self, key: "Selection") -> T_StandardArray:
        return self.array.__getitem__(key)


def _key_to_index_tuple(selection: "Selection") -> Tuple[int, ...]:
    if isinstance(selection, (slice, Integral)):
        selection = (selection,)
    assert all(isinstance(s, (slice, Integral)) for s in selection)  # type: ignore[union-attr]
    sel = []
    for s in selection:  # type: ignore[union-attr]
        if isinstance(s, Integral):
            sel.append(int(s))
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
    shape: T_Shape, *, dtype: T_DType, chunks: T_RegularChunks, **kwargs: Any
) -> VirtualEmptyArray:
    return VirtualEmptyArray(shape, dtype, chunks, **kwargs)


def virtual_full(
    shape: T_Shape,
    fill_value: Any,
    *,
    dtype: T_DType,
    chunks: T_RegularChunks,
    **kwargs: Any,
) -> VirtualFullArray:
    return VirtualFullArray(shape, dtype, chunks, fill_value, **kwargs)


def virtual_offsets(shape: T_Shape) -> VirtualOffsetsArray:
    return VirtualOffsetsArray(shape)


def virtual_in_memory(
    array: T_StandardArray,
    chunks: T_RegularChunks,
) -> VirtualInMemoryArray:
    return VirtualInMemoryArray(array, chunks)
