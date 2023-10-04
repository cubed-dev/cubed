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

from .array_api import Array
from .core.array import (
    compute,
    measure_reserved_mem,
    measure_reserved_memory,
    visualize,
)
from .core.gufunc import apply_gufunc
from .core.ops import from_array, from_zarr, map_blocks, store, to_zarr
from .nan_functions import nanmean, nansum
from .runtime.types import Callback, TaskEndEvent
from .spec import Spec

__all__ = [
    "__version__",
    "Callback",
    "Array",
    "Spec",
    "TaskEndEvent",
    "apply_gufunc",
    "compute",
    "from_array",
    "from_zarr",
    "map_blocks",
    "measure_reserved_mem",
    "measure_reserved_memory",
    "nanmean",
    "nansum",
    "store",
    "to_zarr",
    "visualize",
]
