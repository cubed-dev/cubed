from typing import Iterable

import numpy as np
import zarr
from dask.array.core import normalize_chunks
from zarr.util import normalize_shape

from cubed.core import Array, Plan, gensym, map_blocks, new_temp_store, new_temp_zarr
from cubed.utils import to_chunksize


def _arange(a, size):
    start = a[0]
    return np.arange(start * size, (start + 1) * size)


def arange(
    start, /, stop=None, step=1, *, dtype=None, device=None, chunks="auto", spec=None
):
    # TODO: implement step
    # TODO: support array length that isn't a multiple of chunks
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
    out = map_blocks(_arange, out, dtype=dtype, chunks=chunks, size=chunksize)
    return out


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
    target[...] = a

    plan = Plan(name, "asarray", target, spec)
    return Array(name, target, plan)


def ones(shape, *, dtype=None, device=None, chunks="auto", spec=None):
    # write to zarr
    # note that write_empty_chunks=False means no chunks are written to disk, so it is very efficient to create large arrays
    shape = normalize_shape(shape)
    chunksize = to_chunksize(normalize_chunks(chunks, shape=shape, dtype=dtype))
    name = gensym()
    store = new_temp_store(name=name, spec=spec)
    target = zarr.ones(
        shape,
        store=store,
        dtype=dtype,
        chunks=chunksize,
        write_empty_chunks=False,
    )

    plan = Plan(name, "ones", target, spec)
    return Array(name, target, plan)
