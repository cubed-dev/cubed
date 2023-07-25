import itertools
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from cubed.primitive.types import CubedCopySpec, CubedPipeline
from cubed.storage.zarr import open_if_lazy_zarr_array

from .utils import gensym


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


def already_computed(node_dict: Dict[str, Any], resume: Optional[bool] = None) -> bool:
    """
    Return True if the array for a node doesn't have a pipeline to compute it,
    or it has already been computed (all chunks are present).
    """
    pipeline = node_dict.get("pipeline", None)
    if pipeline is None:
        return True

    target = node_dict.get("target", None)
    if resume and target is not None:
        target = open_if_lazy_zarr_array(target)
        # this check can be expensive since it has to list the directory to find nchunks_initialized
        if target.ndim > 0 and target.nchunks_initialized == target.nchunks:
            return True

    return False
