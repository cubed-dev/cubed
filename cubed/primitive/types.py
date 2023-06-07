from dataclasses import dataclass
from typing import Any, Iterable, Optional

from cubed.storage.zarr import open_if_lazy_zarr_array
from cubed.vendor.rechunker.types import Config, Stage


@dataclass(frozen=True)
class CubedPipeline:
    """Generalisation of rechunker ``Pipeline`` with extra attributes."""

    stages: Iterable[Stage]
    config: Config
    target_array: Any
    intermediate_array: Optional[Any]
    projected_mem: int
    num_tasks: int


class CubedArrayProxy:
    """Generalisation of rechunker ``ArrayProxy`` with support for ``LazyZarrArray``."""

    def __init__(self, array, chunks):
        self.array = array
        self.chunks = chunks

    def open(self):
        return open_if_lazy_zarr_array(self.array)
