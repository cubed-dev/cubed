from itertools import product
from operator import mul

import numpy as np
from dask.array.core import normalize_chunks
from toolz import reduce

from cubed.utils import to_chunksize

from .zarr_view import Translator, ZarrArrayView


def reshape_chunks(source, shape, chunks):
    """Reshape an array by reshaping its chunks.

    This operation provides only a subset of a regular reshape operation since the target array
    must have the same number of chunks as the source.

    The returned array is a view of a Zarr array. No extra storage or computation is required.

    Parameters
    ----------
    source : Zarr array
    shape : tuple
        The array shape to be reshaped to.
    chunks : tuple
        The chunks for the reshaped array.
    Returns
    -------
    target:  Array for the Zarr array output
    """

    if reduce(mul, shape, 1) != source.size:
        raise ValueError("total size of new array must be unchanged")

    source_chunkset = normalize_chunks(
        source.chunks, shape=source.shape, dtype=source.dtype
    )
    target_chunkset = normalize_chunks(chunks, shape=shape, dtype=source.dtype)

    chunk_shapes = list(product(*target_chunkset))
    if len(set(chunk_shapes)) != 1:
        raise ValueError(
            "cannot reshape chunks unless all chunks are the same and exactly divide array"
        )

    chunks = to_chunksize(target_chunkset)

    return ZarrArrayView(
        source, ReshapeViewTranslator(shape, chunks, source_chunkset, target_chunkset)
    )


class ReshapeViewTranslator(Translator):
    def __init__(self, shape, chunks, source_chunkset, target_chunkset):
        self.shape = shape
        self.chunks = chunks
        self.source_chunkset = source_chunkset
        self.target_chunkset = target_chunkset

        self.source_keys = list(product(*[range(len(c)) for c in self.source_chunkset]))
        self.target_keys = list(product(*[range(len(c)) for c in self.target_chunkset]))

    def to_source_chunk_coords(self, target_chunk_coords):
        # special case for reshaping from (1,) to ()
        if self.target_chunkset == ():
            return target_chunk_coords

        return self.source_keys[self.target_keys.index(target_chunk_coords)]

    def to_target_chunk(self, source_chunk):
        # TODO: this is where we make the assumption that all chunks are the same (even end chunks)
        return np.reshape(source_chunk, self.chunks)
