from math import ceil, prod

import numpy as np
import xarray_beam as xbeam
from dask.array.core import normalize_chunks

from cubed.core.plan import new_temp_zarr
from cubed.runtime.utils import gensym
from cubed.utils import chunk_memory, to_chunksize


def rechunk(source, target_chunks, max_mem, target_store, temp_store=None, name=None):
    """Rechunk a Zarr array to have target_chunks.

    Parameters
    ----------
    source : Zarr array
    target_chunks : tuple
        The desired chunks of the array after rechunking.
    max_mem : int
        Maximum memory allowed for a single task of this operation, measured in bytes
    target_store : str
        Path to output Zarr store.
    temp_store : str, optional
        Path to temporary store for intermediate data.

    Returns
    -------
    pipeline:  Pipeline to run the operation
    pcollection: PCollection representing the rechunked array
    target:  Array for the Zarr array output
    required_mem: minimum memory required per-task, in bytes
    """

    # don't give the full max_mem to rechunker, since it doesn't take
    # compressed copies into account
    # instead, force it to use no more than a single source or target chunk
    # (whichever is larger)
    # this may mean an intermediate copy is needed, but ensures that memory is controlled
    dtype = source.dtype  # dtype doesn't change
    adjusted_max_mem = max(
        chunk_memory(dtype, to_chunksize(source.chunks)),
        chunk_memory(dtype, target_chunks),
    )
    if adjusted_max_mem > max_mem:
        raise ValueError(
            f"Source/target chunk memory ({adjusted_max_mem}) exceeds max_mem ({max_mem})"
        )

    shape = source.shape
    source_chunksize = to_chunksize(
        normalize_chunks(source.chunks, shape=shape, dtype=dtype)
    )
    target_chunksize = to_chunksize(
        normalize_chunks(target_chunks, shape=shape, dtype=dtype)
    )

    dims = tuple(f"dim_{i}" for i in range(source.ndim))
    dim_sizes = dict(zip(dims, source.shape))
    source_xchunks = dict(zip(dims, source_chunksize))
    target_xchunks = dict(zip(dims, target_chunksize))
    itemsize = np.dtype(dtype).itemsize

    pipeline = source.plan.beam_pipeline
    pcollection = source.plan.pcollections[source.name]

    ptransform = xbeam.Rechunk(
        dim_sizes=dim_sizes,
        source_chunks=source_xchunks,
        target_chunks=target_xchunks,
        itemsize=itemsize,
        max_mem=max_mem,
    )

    pcollection = pcollection | gensym("rechunk") >> ptransform

    target = new_temp_zarr(
        source.shape, dtype, target_chunksize, name=name, spec=source.spec
    )

    # calculate memory requirement
    # memory for {compressed, uncompressed} x {input, output} array chunk/selection
    required_mem = adjusted_max_mem * 4

    num_tasks = 1  # TODO

    return pipeline, pcollection, target, required_mem, num_tasks


def total_chunks(shape, chunks):
    # cf rechunker's chunk_keys
    return prod(ceil(s / c) for s, c in zip(shape, chunks))
