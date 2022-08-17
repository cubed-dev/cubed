from typing import Iterable

import numpy as np
import zarr
from dask.array.core import normalize_chunks
from zarr.util import normalize_shape

from cubed.core import (
    CoreArray,
    Plan,
    gensym,
    map_blocks,
    new_temp_store,
    new_temp_zarr,
)
from cubed.core.ops import map_direct
from cubed.utils import to_chunksize


def arange(
    start, /, stop=None, step=1, *, dtype=None, device=None, chunks="auto", spec=None
):
    if stop is None:
        start, stop = 0, start
    num = int(max(np.ceil((stop - start) / step), 0))
    if dtype is None:
        dtype = np.arange(start, stop, step * num if num else step).dtype
    chunks = normalize_chunks(chunks, shape=(num,), dtype=dtype)
    chunksize = chunks[0][0]
    numblocks = len(chunks[0])
    # create small array of block numbers
    out = asarray(np.arange(numblocks), chunks=(1,), spec=spec)
    # then map each block to partial arange
    out = map_blocks(
        _arange,
        out,
        dtype=dtype,
        chunks=chunks,
        size=chunksize,
        start=start,
        stop=stop,
        step=step,
    )
    return out


def _arange(a, size, start, stop, step):
    i = a[0]
    blockstart = start + (i * size * step)
    blockstop = start + ((i + 1) * size * step)
    return np.arange(blockstart, min(blockstop, stop), step)


def asarray(obj, /, *, dtype=None, device=None, copy=None, chunks="auto", spec=None):
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
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    # write to zarr
    chunksize = to_chunksize(normalize_chunks(chunks, shape=a.shape, dtype=dtype))
    name = gensym()
    target = new_temp_zarr(a.shape, dtype, chunksize, name=name, spec=spec)
    if a.size > 0:
        target[...] = a

    plan = Plan(name, "asarray", target)
    return CoreArray._new(name, target, spec, plan)


def empty(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    if dtype is None:
        dtype = np.float64
    return full(shape, None, dtype=dtype, device=device, chunks=chunks, spec=spec)


def empty_like(x, /, *, dtype=None, device=None, chunks=None, spec=None):
    return empty(**_like_args(x, dtype, device, chunks, spec))


def eye(
    n_rows, n_cols=None, /, *, k=0, dtype=None, device=None, chunks="auto", spec=None
):
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
        extra_required_mem=0,
        spec=spec,
        k=k,
        chunksize=chunksize,
    )


def _eye(x, *arrays, k=None, chunksize=None, block_id=None):
    i, j = block_id
    bk = (j - i) * chunksize
    if bk - chunksize <= k <= bk + chunksize:
        return np.eye(x.shape[0], x.shape[1], k=k - bk, dtype=x.dtype)
    else:
        return np.zeros_like(x)


def full(shape, fill_value, *, dtype=None, device=None, chunks="auto", spec=None):
    # write to zarr
    # note that write_empty_chunks=False means no chunks are written to disk, so it is very efficient to create large arrays
    shape = normalize_shape(shape)
    if dtype is None:
        if isinstance(fill_value, int):
            dtype = np.int64
        elif isinstance(fill_value, float):
            dtype = np.float64
        elif isinstance(fill_value, bool):
            dtype = np.bool_
        else:
            raise TypeError("Invalid input to full")
    chunksize = to_chunksize(normalize_chunks(chunks, shape=shape, dtype=dtype))
    name = gensym()
    store = new_temp_store(name=name, spec=spec)
    target = zarr.full(
        shape,
        fill_value,
        store=store,
        dtype=dtype,
        chunks=chunksize,
        write_empty_chunks=False,
    )

    plan = Plan(name, "full", target)
    return CoreArray._new(name, target, spec, plan)


def full_like(x, /, fill_value, *, dtype=None, device=None, chunks=None, spec=None):
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
):
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
        extra_required_mem=0,
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
    return np.linspace(
        blockstart, blockstop, bs, endpoint=endpoint, dtype=linspace_dtype
    )


def meshgrid(*arrays, indexing="xy"):
    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("meshgrid inputs must all have the same dtype")

    from cubed.array_api.manipulation_functions import broadcast_arrays

    # based on dask
    if indexing not in ("ij", "xy"):
        raise ValueError("`indexing` must be `'ij'` or `'xy'`")

    if indexing == "xy" and len(arrays) > 1:
        arrays = list(arrays)
        arrays[0], arrays[1] = arrays[1], arrays[0]

    grid = []
    for i in range(len(arrays)):
        s = len(arrays) * [None]
        s[i] = slice(None)
        s = tuple(s)

        r = arrays[i][s]

        grid.append(r)

    grid = broadcast_arrays(*grid)

    if indexing == "xy" and len(arrays) > 1:
        grid[0], grid[1] = grid[1], grid[0]

    return grid


def ones(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    if dtype is None:
        dtype = np.float64
    return full(shape, 1, dtype=dtype, device=device, chunks=chunks, spec=spec)


def ones_like(x, /, *, dtype=None, device=None, chunks=None, spec=None):
    return ones(**_like_args(x, dtype, device, chunks, spec))


def tril(x, /, *, k=0):
    from cubed.array_api.searching_functions import where

    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for tril")

    mask = _tri_mask(*x.shape[-2:], k, x.chunks[-2:], x.spec)
    return where(mask, x, zeros_like(x))


def triu(x, /, *, k=0):
    from cubed.array_api.searching_functions import where

    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for triu")

    mask = _tri_mask(*x.shape[-2:], k - 1, x.chunks[-2:], x.spec)
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


def zeros(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    if dtype is None:
        dtype = np.float64
    return full(shape, 0, dtype=dtype, device=device, chunks=chunks, spec=spec)


def zeros_like(x, /, *, dtype=None, device=None, chunks=None, spec=None):
    return zeros(**_like_args(x, dtype, device, chunks, spec))


def _like_args(x, dtype=None, device=None, chunks=None, spec=None):
    if dtype is None:
        dtype = x.dtype
    if chunks is None:
        chunks = x.chunks
    if spec is None:
        spec = x.spec
    return dict(shape=x.shape, dtype=dtype, device=device, chunks=chunks, spec=spec)
