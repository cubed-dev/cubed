from itertools import product
from operator import mul

import numpy as np
from dask.array.core import normalize_chunks
from toolz import reduce
from zarr.core import Array

from cubed.utils import to_chunksize


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

    original_chunkset = normalize_chunks(
        source.chunks, shape=source.shape, dtype=source.dtype
    )
    new_chunkset = normalize_chunks(chunks, shape=shape, dtype=source.dtype)

    chunk_shapes = list(product(*new_chunkset))
    if len(set(chunk_shapes)) != 1:
        raise ValueError(
            "cannot reshape chunks unless all chunks are the same and exactly divide array"
        )

    chunks = to_chunksize(new_chunkset)

    store = source._store
    chunk_store = source._chunk_store
    path = source._path
    read_only = True
    synchronizer = source._synchronizer
    cache_metadata = True
    cache_attrs = True
    partial_decompress = False
    write_empty_chunks = True

    a = ReshapedArray(
        store=store,
        path=path,
        read_only=read_only,
        chunk_store=chunk_store,
        synchronizer=synchronizer,
        cache_metadata=cache_metadata,
        cache_attrs=cache_attrs,
        partial_decompress=partial_decompress,
        write_empty_chunks=write_empty_chunks,
        shape=shape,
        chunks=chunks,
        original_chunks=source.chunks,
        original_chunkset=original_chunkset,
        new_chunkset=new_chunkset,
    )
    return a


class ReshapedArray(Array):
    # TODO: make this less brittle to changes in Zarr version (which may change init params)
    def __init__(
        self,
        store,
        path,
        read_only,
        chunk_store,
        synchronizer,
        cache_metadata,
        cache_attrs,
        partial_decompress,
        write_empty_chunks,
        shape,
        chunks,
        original_chunks,
        original_chunkset,
        new_chunkset,
    ):
        super().__init__(
            store,
            path,
            read_only,
            chunk_store,
            synchronizer,
            cache_metadata,
            cache_attrs,
            partial_decompress,
            write_empty_chunks,
        )
        self._is_view = True
        self._shape = shape
        self._chunks = chunks
        self._original_chunks = original_chunks
        self._original_chunkset = original_chunkset
        self._new_chunkset = new_chunkset

    def __getstate__(self):
        return (
            self._store,
            self._path,
            self._read_only,
            self._chunk_store,
            self._synchronizer,
            self._cache_metadata,
            self._attrs.cache,
            self._partial_decompress,
            self._write_empty_chunks,
            self._shape,
            self._chunks,
            self._original_chunks,
            self._original_chunkset,
            self._new_chunkset,
        )

    def _chunk_key(self, chunk_coords):
        # change chunk_coords to original (not reshaped) values

        # TODO: find better way than computing cartesian product of all keys each time
        in_keys = list(product(*[range(len(c)) for c in self._original_chunkset]))
        out_keys = list(product(*[range(len(c)) for c in self._new_chunkset]))
        chunk_coords = in_keys[out_keys.index(chunk_coords)]
        return super()._chunk_key(chunk_coords)

    def _decode_chunk(self, cdata, start=None, nitems=None, expected_shape=None):
        # reshape the decoded chunk to the reshaped array chunk size
        if expected_shape is not None:
            raise NotImplementedError("ReshapedArray does not support expected_shape")
        chunk = super()._decode_chunk(
            cdata, start=start, nitems=nitems, expected_shape=self._original_chunks
        )
        # TODO: this is where we make the assumption that all chunks are the same (even end chunks)
        return np.reshape(chunk, self._chunks)
