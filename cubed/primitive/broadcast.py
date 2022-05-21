import numpy as np
from zarr.core import Array


def broadcast_to(source, shape, chunks=None):
    """Broadcast an array to a specified shape.

    The broadcast array will have a chunksize of 1 in the broadcast dimensions, unless the `chunks`
    parameter is specified.

    The returned array is a view of a Zarr array. No extra storage or computation is required.

    Parameters
    ----------
    source : Zarr array
    shape : tuple
        The array shape to be broadcast to.
    chunks : tuple, optional
        If provided, then the result will use these chunks instead of the same
        chunks as the source array. Setting chunks explicitly as part of
        broadcast_to is more efficient than rechunking afterwards. Chunks are
        only allowed to differ from the original shape along dimensions that
        are new on the result or have size 1 the input array.
    Returns
    -------
    target:  Array for the Zarr array output
    """
    ndim_new = len(shape) - source.ndim
    if ndim_new < 0 or any(
        new != old for new, old in zip(shape[ndim_new:], source.shape) if old != 1
    ):
        raise ValueError(f"cannot broadcast shape {source.shape} to shape {shape}")

    store = source._store
    chunk_store = source._chunk_store
    path = source._path
    read_only = True
    synchronizer = source._synchronizer
    cache_metadata = True
    cache_attrs = True
    partial_decompress = False
    write_empty_chunks = True

    if chunks is None:
        chunks = (1,) * ndim_new + source.chunks
    else:
        for old_bd, new_bd in zip(source.chunks, chunks[ndim_new:]):
            if old_bd != new_bd and old_bd != 1:
                raise ValueError(
                    "cannot broadcast chunks %s to chunks %s: "
                    "new chunks must either be along a new "
                    "dimension or a dimension of size 1" % (source.chunks, chunks)
                )

    bds = [i for i, s in enumerate(source.shape) if s != shape[ndim_new:][i]]

    a = BroadcastArray(
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
        bds=bds,
        ndim_new=ndim_new,
    )
    return a


class BroadcastArray(Array):
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
        bds,
        ndim_new,
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
        self._bds = bds
        self._ndim_new = ndim_new

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
            self._bds,
            self._ndim_new,
        )

    def _chunk_key(self, chunk_coords):
        # change chunk_coords to original (pre-broadcast) values
        chunk_coords = tuple(
            0 if i in self._bds else c
            for i, c in enumerate(chunk_coords[self._ndim_new :])
        )
        return super()._chunk_key(chunk_coords)

    def _decode_chunk(self, cdata, start=None, nitems=None, expected_shape=None):
        # broadcast the decoded chunk to the broadcast array chunk size
        if expected_shape is not None:
            raise NotImplementedError("BroadcastArray does not support expected_shape")
        chunk = super()._decode_chunk(
            cdata, start=start, nitems=nitems, expected_shape=self._original_chunks
        )
        return np.broadcast_to(chunk, shape=self._chunks)
