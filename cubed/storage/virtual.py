from numbers import Integral

import numpy as np
import zarr
from zarr.indexing import is_slice

from cubed.types import T_Shape


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
            return np.array(0, dtype=self.dtype)
        return np.ravel_multi_index(_key_to_index_tuple(key), self.shape)


def _key_to_index_tuple(selection):
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


def virtual_offsets(shape: T_Shape) -> VirtualOffsetsArray:
    return VirtualOffsetsArray(shape)
