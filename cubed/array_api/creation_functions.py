import math
from typing import TYPE_CHECKING, Iterable, List

import numpy as np
from zarr.util import normalize_shape

from cubed.backend_array_api import namespace as nxp
from cubed.core import Plan, gensym
from cubed.core.ops import map_direct
from cubed.core.plan import new_temp_path
from cubed.storage.virtual import virtual_empty, virtual_full, virtual_offsets
from cubed.storage.zarr import lazy_from_array
from cubed.utils import to_chunksize
from cubed.vendor.dask.array.core import normalize_chunks

if TYPE_CHECKING:
    from .array_object import Array


def arange(
    start, /, stop=None, step=1, *, dtype=None, device=None, chunks="auto", spec=None
) -> "Array":
    if stop is None:
        start, stop = 0, start
    num = int(max(math.ceil((stop - start) / step), 0))
    shape = (num,)
    if dtype is None:
        dtype = nxp.arange(start, stop, step * num if num else step).dtype
    chunks = normalize_chunks(chunks, shape=(num,), dtype=dtype)
    chunksize = chunks[0][0]

    return map_direct(
        _arange,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_projected_mem=0,
        spec=spec,
        size=chunksize,
        start=start,
        stop=stop,
        step=step,
        arange_dtype=dtype,
    )


def _arange(x, *arrays, size, start, stop, step, arange_dtype, block_id=None):
    i = block_id[0]
    blockstart = start + (i * size * step)
    blockstop = start + ((i + 1) * size * step)
    return nxp.arange(blockstart, min(blockstop, stop), step, dtype=arange_dtype)


def asarray(
    obj, /, *, dtype=None, device=None, copy=None, chunks="auto", spec=None
) -> "Array":
    a = obj
    from cubed.array_api.array_object import Array

    # from dask.asarray
    if isinstance(a, Array):
        return a
    elif type(a).__module__.split(".")[0] == "xarray" and hasattr(
        a, "data"
    ):  # pragma: no cover
        return asarray(a.data)
    elif not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = nxp.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    # write to zarr
    chunksize = to_chunksize(normalize_chunks(chunks, shape=a.shape, dtype=dtype))
    name = gensym()
    zarr_path = new_temp_path(name=name, spec=spec)
    target = lazy_from_array(a, dtype=dtype, chunks=chunksize, store=zarr_path)

    plan = Plan._new(name, "asarray", target)
    return Array(name, target, spec, plan)


def empty(shape, *, dtype=None, device=None, chunks="auto", spec=None) -> "Array":
    shape = normalize_shape(shape)
    return empty_virtual_array(
        shape, dtype=dtype, device=device, chunks=chunks, spec=spec, hidden=False
    )


def empty_like(x, /, *, dtype=None, device=None, chunks=None, spec=None) -> "Array":
    return empty(**_like_args(x, dtype, device, chunks, spec))


def empty_virtual_array(
    shape, *, dtype=None, device=None, chunks="auto", spec=None, hidden=True
) -> "Array":
    if dtype is None:
        dtype = np.float64

    chunksize = to_chunksize(normalize_chunks(chunks, shape=shape, dtype=dtype))
    name = gensym()
    target = virtual_empty(shape, dtype=dtype, chunks=chunksize)

    from .array_object import Array

    plan = Plan._new(name, "empty", target, hidden=hidden)
    return Array(name, target, spec, plan)


def eye(
    n_rows, n_cols=None, /, *, k=0, dtype=None, device=None, chunks="auto", spec=None
) -> "Array":
    if n_cols is None:
        n_cols = n_rows
    if dtype is None:
        dtype = np.float64

    shape = (n_rows, n_cols)
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    chunksize = to_chunksize(chunks)[0]

    return map_direct(
        _eye,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_projected_mem=0,
        spec=spec,
        k=k,
        chunksize=chunksize,
    )


def _eye(x, *arrays, k=None, chunksize=None, block_id=None):
    i, j = block_id
    bk = (j - i) * chunksize
    if bk - chunksize <= k <= bk + chunksize:
        return nxp.eye(x.shape[0], x.shape[1], k=k - bk, dtype=x.dtype)
    else:
        return nxp.zeros_like(x)


def full(
    shape, fill_value, *, dtype=None, device=None, chunks="auto", spec=None
) -> "Array":
    shape = normalize_shape(shape)
    if dtype is None:
        # check bool first since True/False are instances of int and float
        if isinstance(fill_value, bool):
            dtype = np.bool_
        elif isinstance(fill_value, int):
            dtype = np.int64
        elif isinstance(fill_value, float):
            dtype = np.float64
        else:
            raise TypeError("Invalid input to full")
    chunksize = to_chunksize(normalize_chunks(chunks, shape=shape, dtype=dtype))
    name = gensym()
    target = virtual_full(shape, fill_value, dtype=dtype, chunks=chunksize)

    from .array_object import Array

    plan = Plan._new(name, "full", target)
    return Array(name, target, spec, plan)


def offsets_virtual_array(shape, spec=None) -> "Array":
    name = gensym()
    target = virtual_offsets(shape)

    from .array_object import Array

    plan = Plan._new(name, "block_ids", target, hidden=True)
    return Array(name, target, spec, plan)


def full_like(
    x, /, fill_value, *, dtype=None, device=None, chunks=None, spec=None
) -> "Array":
    return full(fill_value=fill_value, **_like_args(x, dtype, device, chunks, spec))


def linspace(
    start,
    stop,
    /,
    num,
    *,
    dtype=None,
    device=None,
    endpoint=True,
    chunks="auto",
    spec=None,
) -> "Array":
    range_ = stop - start
    div = (num - 1) if endpoint else num
    if div == 0:
        div = 1
    step = float(range_) / div
    shape = (num,)
    if dtype is None:
        dtype = np.float64
    chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)
    chunksize = chunks[0][0]

    if num == 0:
        return asarray(0.0, dtype=dtype, spec=spec)

    return map_direct(
        _linspace,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_projected_mem=0,
        spec=spec,
        size=chunksize,
        start=start,
        step=step,
        endpoint=endpoint,
        linspace_dtype=dtype,
    )


def _linspace(x, *arrays, size, start, step, endpoint, linspace_dtype, block_id=None):
    bs = x.shape[0]
    i = block_id[0]
    adjusted_bs = bs - 1 if endpoint else bs
    blockstart = start + (i * size * step)
    blockstop = blockstart + (adjusted_bs * step)
    return nxp.linspace(
        blockstart, blockstop, bs, endpoint=endpoint, dtype=linspace_dtype
    )


def meshgrid(*arrays, indexing="xy") -> List["Array"]:
    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("meshgrid inputs must all have the same dtype")

    from cubed.array_api.manipulation_functions import broadcast_arrays

    # based on dask
    if indexing not in ("ij", "xy"):
        raise ValueError("`indexing` must be `'ij'` or `'xy'`")

    arrs = list(arrays)
    if indexing == "xy" and len(arrs) > 1:
        arrs[0], arrs[1] = arrs[1], arrs[0]

    grid = []
    for i in range(len(arrs)):
        s = len(arrs) * [None]
        s[i] = slice(None)  # type: ignore[call-overload]

        r = arrs[i][tuple(s)]

        grid.append(r)

    grid = list(broadcast_arrays(*grid))

    if indexing == "xy" and len(arrs) > 1:
        grid[0], grid[1] = grid[1], grid[0]

    return grid


def ones(shape, *, dtype=None, device=None, chunks="auto", spec=None) -> "Array":
    if dtype is None:
        dtype = np.float64
    return full(shape, 1, dtype=dtype, device=device, chunks=chunks, spec=spec)


def ones_like(x, /, *, dtype=None, device=None, chunks=None, spec=None) -> "Array":
    return ones(**_like_args(x, dtype, device, chunks, spec))


def tril(x, /, *, k=0) -> "Array":
    from cubed.array_api.searching_functions import where

    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for tril")

    mask = _tri_mask(x.shape[-2], x.shape[-1], k, x.chunks[-2:], x.spec)
    return where(mask, x, zeros_like(x))


def triu(x, /, *, k=0) -> "Array":
    from cubed.array_api.searching_functions import where

    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for triu")

    mask = _tri_mask(x.shape[-2], x.shape[-1], k - 1, x.chunks[-2:], x.spec)
    return where(mask, zeros_like(x), x)


def _tri_mask(N, M, k, chunks, spec):
    from cubed.array_api.elementwise_functions import greater_equal
    from cubed.array_api.manipulation_functions import expand_dims

    # based on dask
    chunks = normalize_chunks(chunks, shape=(N, M))

    # TODO: use min_int for arange dtype
    m = greater_equal(
        expand_dims(arange(N, chunks=chunks[0][0], spec=spec), axis=1),
        arange(-k, M - k, chunks=chunks[1][0], spec=spec),
    )

    return m


def zeros(shape, *, dtype=None, device=None, chunks="auto", spec=None) -> "Array":
    if dtype is None:
        dtype = np.float64
    return full(shape, 0, dtype=dtype, device=device, chunks=chunks, spec=spec)


def zeros_like(x, /, *, dtype=None, device=None, chunks=None, spec=None) -> "Array":
    return zeros(**_like_args(x, dtype, device, chunks, spec))


def _like_args(x, dtype=None, device=None, chunks=None, spec=None):
    if dtype is None:
        dtype = x.dtype
    if chunks is None:
        chunks = x.chunks
    if spec is None:
        spec = x.spec
    return dict(shape=x.shape, dtype=dtype, device=device, chunks=chunks, spec=spec)
