# Suppress numpy.array_api experimental warning
import sys
import warnings

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=UserWarning)

from importlib.metadata import version as _version

try:
    __version__ = _version("cubed")
except Exception:  # pragma: no cover
    __version__ = "unknown"

from .core import (
    Callback,
    CoreArray,
    Spec,
    TaskEndEvent,
    compute,
    from_array,
    from_zarr,
    map_blocks,
    measure_baseline_memory,
    store,
    to_zarr,
    visualize,
)

__all__ = [
    "__version__",
    "Callback",
    "CoreArray",
    "Spec",
    "TaskEndEvent",
    "compute",
    "from_array",
    "from_zarr",
    "map_blocks",
    "measure_baseline_memory",
    "store",
    "to_zarr",
    "visualize",
]
