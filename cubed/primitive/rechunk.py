from math import ceil, prod
from typing import Any, Dict, List, Optional, Tuple

from cubed.primitive.types import CubedArrayProxy, CubedCopySpec, CubedPipeline
from cubed.runtime.pipeline import spec_to_pipeline
from cubed.storage.zarr import T_ZarrArray, lazy_empty
from cubed.types import T_RegularChunks, T_Shape, T_Store
from cubed.vendor.rechunker.algorithm import rechunking_plan
from cubed.vendor.rechunker.api import _validate_options


def rechunk(
    source: T_ZarrArray,
    target_chunks: T_RegularChunks,
    allowed_mem: int,
    reserved_mem: int,
    target_store: T_Store,
    temp_store: Optional[T_Store] = None,
) -> CubedPipeline:
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

    copy_specs, intermediate, target = _setup_rechunk(
        source=source,
        target_chunks=target_chunks,
        max_mem=rechunker_max_mem,
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

    # we assume that rechunker is going to use all the memory it is allowed to
    projected_mem = allowed_mem
    return spec_to_pipeline(copy_spec, target, projected_mem, num_tasks)


# from rechunker, but simpler since it only has to handle Zarr arrays
def _setup_rechunk(
    source: T_ZarrArray,
    target_chunks: T_RegularChunks,
    max_mem: int,
    target_store: T_Store,
    target_options: Optional[Dict[Any, Any]] = None,
    temp_store: Optional[T_Store] = None,
    temp_options: Optional[Dict[Any, Any]] = None,
) -> Tuple[List[CubedCopySpec], T_ZarrArray, T_ZarrArray]:
    if temp_options is None:
        temp_options = target_options
    target_options = target_options or {}
    temp_options = temp_options or {}

    copy_spec = _setup_array_rechunk(
        source,
        target_chunks,
        max_mem,
        target_store,
        target_options=target_options,
        temp_store_or_group=temp_store,
        temp_options=temp_options,
    )
    intermediate = copy_spec.intermediate.array
    target = copy_spec.write.array
    return [copy_spec], intermediate, target


def _setup_array_rechunk(
    source_array: T_ZarrArray,
    target_chunks: T_RegularChunks,
    max_mem: int,
    target_store_or_group: T_Store,
    target_options: Optional[Dict[Any, Any]] = None,
    temp_store_or_group: Optional[T_Store] = None,
    temp_options: Optional[Dict[Any, Any]] = None,
    name: Optional[str] = None,
) -> CubedCopySpec:
    _validate_options(target_options)
    _validate_options(temp_options)
    shape = source_array.shape
    # source_chunks = (
    #     source_array.chunksize
    #     if isinstance(source_array, dask.array.Array)
    #     else source_array.chunks
    # )
    source_chunks = source_array.chunks
    dtype = source_array.dtype
    itemsize = dtype.itemsize

    if target_chunks is None:
        # this is just a pass-through copy
        target_chunks = source_chunks

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
        store=target_store_or_group,
        **(target_options or {}),
    )

    if read_chunks == write_chunks:
        int_array = None
    else:
        # do intermediate store
        if temp_store_or_group is None:
            raise ValueError(
                "A temporary store location must be provided{}.".format(
                    f" (array={name})" if name else ""
                )
            )
        int_array = lazy_empty(
            shape,
            dtype=dtype,
            chunks=int_chunks,
            store=temp_store_or_group,
            **(target_options or {}),
        )

    read_proxy = CubedArrayProxy(source_array, read_chunks)
    int_proxy = CubedArrayProxy(int_array, int_chunks)
    write_proxy = CubedArrayProxy(target_array, write_chunks)
    return CubedCopySpec(read_proxy, int_proxy, write_proxy)


def total_chunks(shape: T_Shape, chunks: T_RegularChunks) -> int:
    # cf rechunker's chunk_keys
    return prod(ceil(s / c) for s, c in zip(shape, chunks))
