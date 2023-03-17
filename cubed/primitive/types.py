from dataclasses import dataclass
from typing import Any, Iterable

from cubed.vendor.rechunker.types import Config, Stage


@dataclass(frozen=True)
class CubedPipeline:
    """Generalisation of rechunker ``Pipeline`` with extra attributes."""

    stages: Iterable[Stage]
    config: Config
    target_array: Any
    required_mem: int
    num_tasks: int
