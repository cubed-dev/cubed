import warnings

warnings.warn(
    "`cubed.extensions.mem_warn` is deprecated, please use `cubed.diagnostics.mem_warn` instead",
    DeprecationWarning,
)

from cubed.diagnostics.mem_warn import MemoryWarningCallback  # noqa: F401
