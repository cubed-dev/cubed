from math import ceil, prod

from rechunker.api import _setup_rechunk

from cubed.runtime.pipeline import spec_to_pipeline


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

    # rechunker doesn't take account of uncompressed and compressed copies of the
    # input and output array chunk/selection, so adjust appropriately
    adjusted_max_mem = max_mem / 4

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

    num_tasks = total_chunks(copy_spec.write.array.shape, copy_spec.write.chunks)
    if intermediate is not None:
        num_tasks += total_chunks(copy_spec.read.array.shape, copy_spec.read.chunks)

    return spec_to_pipeline(copy_spec, target, max_mem, num_tasks)


def total_chunks(shape, chunks):
    # cf rechunker's chunk_keys
    return prod(ceil(s / c) for s, c in zip(shape, chunks))
