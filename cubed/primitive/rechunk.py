import itertools
import math
from math import ceil, prod
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from cubed.primitive.types import CubedArrayProxy, CubedCopySpec
from cubed.runtime.types import CubedPipeline
from cubed.storage.zarr import T_ZarrArray, lazy_empty
from cubed.types import T_RegularChunks, T_Shape, T_Store
from cubed.vendor.rechunker.algorithm import rechunking_plan

sym_counter = 0


def gensym(name: str) -> str:
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


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


# differs from rechunker's chunk_keys to return a list rather than a tuple, to keep lithops happy
def chunk_keys(
    shape: Tuple[int, ...], chunks: Tuple[int, ...]
) -> Iterator[List[slice]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    ranges = [range(math.ceil(s / c)) for s, c in zip(shape, chunks)]
    for indices in itertools.product(*ranges):
        yield [
            slice(c * i, min(c * (i + 1), s)) for i, s, c in zip(indices, shape, chunks)
        ]


# need a ChunkKeys Iterable instead of a generator, to avoid beam pickle error
class ChunkKeys(Iterable[Tuple[slice, ...]]):
    def __init__(self, shape: Tuple[int, ...], chunks: Tuple[int, ...]):
        self.shape = shape
        self.chunks = chunks

    def __iter__(self):
        return chunk_keys(self.shape, self.chunks)


def copy_read_to_write(chunk_key: Sequence[slice], *, config: CubedCopySpec) -> None:
    # workaround limitation of lithops.utils.verify_args
    if isinstance(chunk_key, list):
        chunk_key = tuple(chunk_key)
    data = np.asarray(config.read.open()[chunk_key])
    config.write.open()[chunk_key] = data


def spec_to_pipeline(
    spec: CubedCopySpec,
    target_array: Any,
    projected_mem: int,
    reserved_mem: int,
    num_tasks: int,
) -> CubedPipeline:
    # typing won't work until we start using numpy types
    shape = spec.read.array.shape  # type: ignore
    return CubedPipeline(
        copy_read_to_write,
        gensym("copy_read_to_write"),
        ChunkKeys(shape, spec.write.chunks),
        spec,
        target_array,
        projected_mem,
        reserved_mem,
        num_tasks,
        spec.write.chunks,
    )
