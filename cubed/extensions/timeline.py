import warnings

warnings.warn(
    "`cubed.extensions.timeline` is deprecated, please use `cubed.diagnostics.timeline` instead",
    DeprecationWarning,
)

from cubed.diagnostics.timeline import TimelineVisualizationCallback  # noqa: F401
