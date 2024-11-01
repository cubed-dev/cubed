from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import obstore as obs
import zarr

from cubed.runtime.types import CubedPipeline
from cubed.storage.zarr import T_ZarrArray, open_if_lazy_zarr_array
from cubed.types import T_RegularChunks
from cubed.vendor.dask.array.core import normalize_chunks


@dataclass(frozen=True)
class PrimitiveOperation:
    """Encapsulates metadata about a ``blockwise`` or ``rechunk`` primitive operation."""

    pipeline: CubedPipeline
    """The pipeline that runs this operation."""

    source_array_names: List[str]
    """The names of the arrays which are inputs to this operation."""

    target_array: Any
    """The array or arrays being computed by this operation."""

    projected_mem: int
    """An upper bound of the memory needed to run a task, in bytes."""

    allowed_mem: int
    """
    The total memory available to a worker for running a task, in bytes.

    This includes any ``reserved_mem`` that has been set.
    """

    reserved_mem: int
    """The memory reserved on a worker for non-data use when running a task, in bytes."""

    num_tasks: int
    """The number of tasks needed to run this operation."""

    fusable: bool = True
    """Whether this operation can be fused with predecessor operations."""

    write_chunks: Optional[T_RegularChunks] = None
    """The chunk size used by this operation."""


class CubedArrayProxy:
    """Generalisation of rechunker ``ArrayProxy`` with support for ``LazyZarrArray``."""

    def __init__(
        self,
        array: T_ZarrArray,
        chunks: T_RegularChunks,
        name: str,
        use_object_store: bool = False,
    ):
        self.array = array
        self.chunks = chunks
        self.normalized_chunks = normalize_chunks(
            chunks, shape=array.shape, dtype=array.dtype
        )
        self.name = name
        self.use_object_store = use_object_store
        if use_object_store:
            Path(array.store).mkdir(parents=True, exist_ok=True)
            self.object_store = obs.store.LocalStore(array.store)

    def open(self) -> zarr.Array:
        return open_if_lazy_zarr_array(self.array)

    def read_chunk(self, coords):
        """Read a chunk from storage keyed by coordinate"""
        if self.use_object_store:
            key = to_obj_store_key(self.name, coords)
            response = obs.get(self.object_store, key)
            a = np.frombuffer(response.bytes(), dtype=self.array.dtype)
            # shape information is lost when we convert a numpy array to bytes and back, so reconstruct
            chunk_shape = tuple(
                ch[i] for ch, i in zip(self.normalized_chunks, coords)
            )  # TODO: naming
            return a.reshape(chunk_shape)
        else:
            from cubed.primitive.blockwise import key_to_slices

            out_chunk_key = key_to_slices(coords, self.array, self.chunks)
            return self.open()[out_chunk_key]

    def write_chunk(self, coords, chunk_array):
        """Write a chunk to storage keyed by coordinate"""
        if self.use_object_store:
            key = to_obj_store_key(self.name, coords)
            obs.put(self.object_store, key, chunk_array.tobytes())
        else:
            from cubed.primitive.blockwise import key_to_slices

            out_chunk_key = key_to_slices(coords, self.array, self.chunks)
            self.open()[out_chunk_key] = chunk_array


def to_obj_store_key(
    array_name: str,
    key: Tuple[int, ...],
) -> str:
    """Convert a chunk index key and array name to an object store key"""
    coords = "/".join(str(c) for c in key)
    return f"{array_name}/{coords}"


@dataclass(frozen=True)
class CubedCopySpec:
    """Generalisation of rechunker ``CopySpec`` with support for ``LazyZarrArray``."""

    read: CubedArrayProxy
    write: CubedArrayProxy


class MemoryModeller:
    """Models peak memory usage for a series of operations."""

    current_mem: int = 0
    peak_mem: int = 0

    def allocate(self, num_bytes):
        self.current_mem += num_bytes
        self.peak_mem = max(self.peak_mem, self.current_mem)

    def free(self, num_bytes):
        self.current_mem -= num_bytes
        self.peak_mem = max(self.peak_mem, self.current_mem)
