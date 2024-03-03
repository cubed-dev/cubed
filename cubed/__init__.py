from importlib.metadata import version as _version

try:
    __version__ = _version("cubed")
except Exception:  # pragma: no cover
    __version__ = "unknown"

from donfig import Config

config = Config(
    "cubed",
    # default spec is local temp dir and a modest amount of memory (200MB, of which 100MB is reserved)
    defaults=[{"spec": {"allowed_mem": 200_000_000, "reserved_mem": 100_000_000}}],
)

from .array_api import Array
from .core.array import compute, measure_reserved_mem, visualize
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
    "config",
    "from_array",
    "from_zarr",
    "map_blocks",
    "measure_reserved_mem",
    "nanmean",
    "nansum",
    "store",
    "to_zarr",
    "visualize",
]
