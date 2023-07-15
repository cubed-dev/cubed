from dataclasses import dataclass
from typing import Any, Iterable, Optional

import zarr

from cubed.storage.zarr import T_ZarrArray, open_if_lazy_zarr_array
from cubed.types import T_RegularChunks
from cubed.vendor.rechunker.types import Config, StageFunction


@dataclass(frozen=True)
class CubedPipeline:
    """Generalisation of rechunker ``Pipeline`` with extra attributes."""

    function: StageFunction
    name: str
    mappable: Iterable
    config: Config
    target_array: Any
    projected_mem: int
    reserved_mem: int
    num_tasks: int
    write_chunks: Optional[T_RegularChunks]


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
