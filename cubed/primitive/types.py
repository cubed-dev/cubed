from dataclasses import dataclass
from typing import Any, List, Optional

import zarr

from cubed.runtime.types import CubedPipeline
from cubed.storage.zarr import T_ZarrArray, open_if_lazy_zarr_array
from cubed.types import T_RegularChunks


@dataclass(frozen=True)
class PrimitiveOperation:
    """Encapsulates metadata about a ``blockwise`` or ``rechunk`` primitive operation."""

    pipeline: CubedPipeline
    """The pipeline that runs this operation."""

    source_array_names: List[str]
    """The names of the arrays which are inputs to this operation."""

    target_array: Any
    """The array being computed by this operation."""

    projected_mem: int
    """An upper bound of the memory needed to run a task, in bytes."""

    allowed_mem: int
    """
    The total memory available to a worker for running a task, in bytes.

    This includes any ``reserved_mem`` that has been set.
    """

    reserved_mem: int
    """The memory reserved on a worker for non-data use when running a task, in bytes."""

    num_tasks: int
    """The number of tasks needed to run this operation."""

    fusable: bool = True
    """Whether this operation should be considered for fusion."""

    write_chunks: Optional[T_RegularChunks] = None
    """The chunk size used by this operation."""


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


class MemoryModeller:
    """Models peak memory usage for a series of operations."""

    current_mem: int = 0
    peak_mem: int = 0

    def allocate(self, num_bytes):
        self.current_mem += num_bytes
        self.peak_mem = max(self.peak_mem, self.current_mem)

    def free(self, num_bytes):
        self.current_mem -= num_bytes
        self.peak_mem = max(self.peak_mem, self.current_mem)
