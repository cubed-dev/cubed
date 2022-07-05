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
    Spec,
    TqdmProgressBar,
    from_zarr,
    map_blocks,
    std_out_err_redirect_tqdm,
    to_zarr,
)

__all__ = [
    "__version__",
    "Callback",
    "Spec",
    "TqdmProgressBar",
    "from_zarr",
    "map_blocks",
    "std_out_err_redirect_tqdm",
    "to_zarr",
]
