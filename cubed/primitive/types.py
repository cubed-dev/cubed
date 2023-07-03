from dataclasses import dataclass
from typing import Any, Optional, Sequence

import zarr

from cubed.storage.zarr import T_ZarrArray, open_if_lazy_zarr_array
from cubed.types import T_RegularChunks
from cubed.vendor.rechunker.types import Config, Stage


@dataclass(frozen=True)
class CubedPipeline:
    """Generalisation of rechunker ``Pipeline`` with extra attributes."""

    stages: Sequence[Stage]
    config: Config
    target_array: Any
    intermediate_array: Optional[Any]
    projected_mem: int
    num_tasks: int


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
    intermediate: CubedArrayProxy
    write: CubedArrayProxy
