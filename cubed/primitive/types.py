from dataclasses import dataclass
from typing import Any, Optional

import zarr

from cubed.runtime.types import CubedPipeline
from cubed.storage.zarr import T_ZarrArray, open_if_lazy_zarr_array
from cubed.types import T_RegularChunks


@dataclass(frozen=True)
class PrimitiveOperation:
    """Encapsulates metadata about a ``blockwise`` or ``rechunk`` primitive operation."""

    pipeline: CubedPipeline
    target_array: Any
    projected_mem: int
    reserved_mem: int
    num_tasks: int
    write_chunks: Optional[T_RegularChunks] = None


class CubedArrayProxy:
    """Generalisation of rechunker ``ArrayProxy`` with support for ``LazyZarrArray``."""

    def __init__(self, array: T_ZarrArray, chunks: T_RegularChunks):
        self.array = array
        self.chunks = chunks

    def open(self) -> zarr.Array:
        return open_if_lazy_zarr_array(self.array)


@dataclass(frozen=True)
class CubedCopySpec:
    """Generalisation of rechunker ``CopySpec`` with support for ``LazyZarrArray``."""

    read: CubedArrayProxy
    write: CubedArrayProxy
