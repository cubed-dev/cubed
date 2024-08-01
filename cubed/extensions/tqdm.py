import warnings

warnings.warn(
    "`cubed.extensions.tqdm` is deprecated, please use `cubed.diagnostics.tqdm` instead",
    DeprecationWarning,
)

from cubed.diagnostics.tqdm import (  # noqa: F401
    TqdmProgressBar,
    std_out_err_redirect_tqdm,
)
