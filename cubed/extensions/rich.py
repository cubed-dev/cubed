import warnings

warnings.warn(
    "`cubed.extensions.rich` is deprecated, please use `cubed.diagnostics.rich` instead",
    DeprecationWarning,
)

from cubed.diagnostics.rich import RichProgressBar  # noqa: F401
