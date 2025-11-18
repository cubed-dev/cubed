from cubed.array_api.utility_functions import any as cubed_any
from cubed.backend_array_api import namespace as nxp
from cubed.core import blockwise


def isin(x1, x2, /, *, invert=False):
    # based on dask isin

    x1_axes = tuple(range(x1.ndim))
    x2_axes = tuple(i + x1.ndim for i in range(x2.ndim))
    mapped = blockwise(
        _isin,
        x1_axes + x2_axes,
        x1,
        x1_axes,
        x2,
        x2_axes,
        dtype=nxp.bool,
        adjust_chunks={axis: lambda _: 1 for axis in x2_axes},
    )

    result = cubed_any(mapped, axis=x2_axes)
    if invert:
        result = ~result
    return result


def _isin(a1, a2):
    a1_flattened = nxp.reshape(a1, (-1,))
    values = nxp.isin(a1_flattened, a2)
    return nxp.reshape(values, a1.shape + (1,) * a2.ndim)
