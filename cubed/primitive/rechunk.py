from math import ceil, prod

from rechunker.api import _setup_rechunk

from cubed.runtime.pipeline import spec_to_pipeline
from cubed.utils import chunk_memory


def rechunk(source, target_chunks, max_mem, target_store, temp_store=None):
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
        chunk_memory(dtype, source.chunks),
        chunk_memory(dtype, target_chunks),
    )
    if adjusted_max_mem > max_mem:
        raise ValueError(
            f"Source/target chunk memory ({adjusted_max_mem}) exceeds max_mem ({max_mem})"
        )

    copy_specs, intermediate, target = _setup_rechunk(
        source=source,
        target_chunks=target_chunks,
        max_mem=adjusted_max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )

    # source is a Zarr array, so only a single copy spec
    if len(copy_specs) != 1:  # pragma: no cover
        raise ValueError(f"Source must be a Zarr array, but was {source}")
    copy_spec = copy_specs[0]

    # calculate memory requirement
    # memory for {compressed, uncompressed} x {input, output} array chunk/selection
    required_mem = adjusted_max_mem * 4

    num_tasks = total_chunks(copy_spec.write.array.shape, copy_spec.write.chunks)
    if intermediate is not None:
        num_tasks += total_chunks(copy_spec.read.array.shape, copy_spec.read.chunks)

    return spec_to_pipeline(copy_spec), target, required_mem, num_tasks


def total_chunks(shape, chunks):
    # cf rechunker's chunk_keys
    return prod(ceil(s / c) for s, c in zip(shape, chunks))
