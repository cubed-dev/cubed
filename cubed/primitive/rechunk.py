from math import ceil, prod
from typing import List, Optional, Tuple

from cubed.primitive.types import CubedArrayProxy, CubedCopySpec, CubedPipeline
from cubed.runtime.pipeline import spec_to_pipeline
from cubed.storage.zarr import T_ZarrArray, lazy_empty
from cubed.types import T_RegularChunks, T_Shape, T_Store
from cubed.vendor.rechunker.algorithm import rechunking_plan


def rechunk(
    source: T_ZarrArray,
    target_chunks: T_RegularChunks,
    allowed_mem: int,
    reserved_mem: int,
    target_store: T_Store,
    temp_store: Optional[T_Store] = None,
) -> List[CubedPipeline]:
    """Rechunk a Zarr array to have target_chunks.

    Parameters
    ----------
    source : Zarr array
    target_chunks : tuple
        The desired chunks of the array after rechunking.
    allowed_mem : int
        The memory available to a worker for running a task, in bytes. Includes ``reserved_mem``.
    reserved_mem : int
        The memory reserved on a worker for non-data use when running a task, in bytes
    target_store : str
        Path to output Zarr store.
    temp_store : str, optional
        Path to temporary store for intermediate data.

    Returns
    -------
    CubedPipeline to run the operation
    """

    # rechunker doesn't take account of uncompressed and compressed copies of the
    # input and output array chunk/selection, so adjust appropriately
    rechunker_max_mem = (allowed_mem - reserved_mem) // 4

    # we assume that rechunker is going to use all the memory it is allowed to
    projected_mem = allowed_mem

    read_proxy, int_proxy, write_proxy = _setup_array_rechunk(
        source_array=source,
        target_chunks=target_chunks,
        max_mem=rechunker_max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )

    intermediate = int_proxy.array
    target = write_proxy.array

    if intermediate is None:
        copy_spec = CubedCopySpec(read_proxy, write_proxy)
        num_tasks = total_chunks(write_proxy.array.shape, write_proxy.chunks)
        return [
            spec_to_pipeline(copy_spec, target, projected_mem, reserved_mem, num_tasks)
        ]

    else:
        # break spec into two if there's an intermediate
        copy_spec1 = CubedCopySpec(read_proxy, int_proxy)
        num_tasks = total_chunks(copy_spec1.write.array.shape, copy_spec1.write.chunks)
        pipeline1 = spec_to_pipeline(
            copy_spec1, intermediate, projected_mem, reserved_mem, num_tasks
        )

        copy_spec2 = CubedCopySpec(int_proxy, write_proxy)
        num_tasks = total_chunks(copy_spec2.write.array.shape, copy_spec2.write.chunks)
        pipeline2 = spec_to_pipeline(
            copy_spec2, target, projected_mem, reserved_mem, num_tasks
        )

        return [pipeline1, pipeline2]


# from rechunker, but simpler since it only has to handle Zarr arrays
def _setup_array_rechunk(
    source_array: T_ZarrArray,
    target_chunks: T_RegularChunks,
    max_mem: int,
    target_store: T_Store,
    temp_store: Optional[T_Store] = None,
) -> Tuple[CubedArrayProxy, CubedArrayProxy, CubedArrayProxy]:
    shape = source_array.shape
    source_chunks = source_array.chunks
    dtype = source_array.dtype
    itemsize = dtype.itemsize

    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize,
        max_mem,
    )

    # create target
    shape = tuple(int(x) for x in shape)  # ensure python ints for serialization
    target_chunks = tuple(int(x) for x in target_chunks)
    int_chunks = tuple(int(x) for x in int_chunks)
    write_chunks = tuple(int(x) for x in write_chunks)

    target_array = lazy_empty(
        shape,
        dtype=dtype,
        chunks=target_chunks,
        store=target_store,
    )

    if read_chunks == write_chunks:
        int_array = None
    else:
        # do intermediate store
        if temp_store is None:
            raise ValueError("A temporary store location must be provided.")
        int_array = lazy_empty(
            shape,
            dtype=dtype,
            chunks=int_chunks,
            store=temp_store,
        )

    read_proxy = CubedArrayProxy(source_array, read_chunks)
    int_proxy = CubedArrayProxy(int_array, int_chunks)
    write_proxy = CubedArrayProxy(target_array, write_chunks)
    return read_proxy, int_proxy, write_proxy


def total_chunks(shape: T_Shape, chunks: T_RegularChunks) -> int:
    # cf rechunker's chunk_keys
    return prod(ceil(s / c) for s, c in zip(shape, chunks))
