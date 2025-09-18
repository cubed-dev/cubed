from cubed.array.overlap import map_overlap
from cubed.array_api.creation_functions import asarray
from cubed.array_api.manipulation_functions import concat
from cubed.backend_array_api import namespace as nxp
from cubed.core import reduction
from cubed.utils import normalize_chunks
from cubed.vendor.dask.array.utils import validate_axis


def all(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.size == 0:
        return asarray(True, dtype=x.dtype)
    return reduction(
        x,
        nxp.all,
        axis=axis,
        dtype=nxp.bool,
        keepdims=keepdims,
        split_every=split_every,
    )


def any(x, /, *, axis=None, keepdims=False, split_every=None):
    if x.size == 0:
        return asarray(False, dtype=x.dtype)
    return reduction(
        x,
        nxp.any,
        axis=axis,
        dtype=nxp.bool,
        keepdims=keepdims,
        split_every=split_every,
    )


def diff(x, /, *, axis=-1, n=1, prepend=None, append=None):
    axis = validate_axis(axis, x.ndim)

    if n < 0:
        raise ValueError(f"order of diff must be non-negative, but was {n}")
    if n == 0:
        return x

    combined = []
    if prepend is not None:
        combined.append(prepend)
    combined.append(x)
    if append is not None:
        combined.append(append)
    if len(combined) > 1:
        x = concat(combined, axis=axis, chunks=x.chunksize)

    shape = tuple(s - n if i == axis else s for i, s in enumerate(x.shape))
    chunks = normalize_chunks(x.chunksize, shape, dtype=x.dtype)
    depth = {axis: (0, n)}  # only need look-ahead values for differencing
    return map_overlap(
        nxp.diff,
        x,
        dtype=x.dtype,
        chunks=chunks,
        depth=depth,
        axis=axis,
        n=n,
    )
