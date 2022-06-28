from typing import Iterable

import numpy as np
import zarr
from dask.array.core import normalize_chunks
from zarr.util import normalize_shape

from cubed.core import Array, Plan, gensym, map_blocks, new_temp_store, new_temp_zarr
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
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
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

    plan = Plan(name, "asarray", target, spec)
    return Array(name, target, plan)


def empty(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    return full(shape, None, dtype=dtype, device=device, chunks=chunks, spec=spec)


def empty_like(x, /, *, dtype=None, device=None, chunks=None, spec=None):
    return empty(**_like_args(x, dtype, device, chunks, spec))


def full(shape, fill_value, *, dtype=None, device=None, chunks="auto", spec=None):
    # write to zarr
    # note that write_empty_chunks=False means no chunks are written to disk, so it is very efficient to create large arrays
    shape = normalize_shape(shape)
    if dtype is None:
        dtype = np.float64
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

    plan = Plan(name, "full", target, spec)
    return Array(name, target, plan)


def full_like(x, /, fill_value, *, dtype=None, device=None, chunks=None, spec=None):
    return full(fill_value=fill_value, **_like_args(x, dtype, device, chunks, spec))


def ones(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    return full(shape, 1, dtype=dtype, device=device, chunks=chunks, spec=spec)


def ones_like(x, /, *, dtype=None, device=None, chunks=None, spec=None):
    return ones(**_like_args(x, dtype, device, chunks, spec))


def zeros(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    return full(shape, 0, dtype=dtype, device=device, chunks=chunks, spec=spec)


def zeros_like(x, /, *, dtype=None, device=None, chunks=None, spec=None):
    return zeros(**_like_args(x, dtype, device, chunks, spec))


def _like_args(x, dtype=None, device=None, chunks=None, spec=None):
    if dtype is None:
        dtype = x.dtype
    if chunks is None:
        chunks = x.chunks
    if spec is None:
        spec = x.plan.spec
    return dict(shape=x.shape, dtype=dtype, device=device, chunks=chunks, spec=spec)
