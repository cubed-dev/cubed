import warnings

warnings.warn(
    "`cubed.extensions.history` is deprecated, please use `cubed.diagnostics.history` instead",
    DeprecationWarning,
)

from cubed.diagnostics.history import HistoryCallback  # noqa: F401
