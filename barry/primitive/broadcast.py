from zarr.core import Array

# TODO: support 'chunks' parameter to allow broadcast dimensions to have arbitrary chunk sizes. (This will require work in Zarr.)


def broadcast_to(source, shape):
    """Broadcast an array to a specified shape.

    The broadcast array will have a chunksize of 1 in the broadcast dimensions.

    The returned array is a view of a Zarr array. No extra storage or computation is required.

    Parameters
    ----------
    source : Zarr array
    shape : tuple
        The array shape to be broadcast to.

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

    a = BroadcastArray(
        store=store,
        path=path,
        chunk_store=chunk_store,
        read_only=read_only,
        synchronizer=synchronizer,
        cache_metadata=True,
    )
    a._dtype = source.dtype
    a._shape = shape  # TODO: normalize_shape
    a._chunks = (1,) * ndim_new + source.chunks
    a._bds = [i for i, s in enumerate(source.shape) if s != shape[ndim_new:][i]]
    a._ndim_new = ndim_new

    return a


class BroadcastArray(Array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_view = True

    def _chunk_key(self, chunk_coords):
        # change chunk_coords to pre-broadcast values
        chunk_coords = tuple(
            0 if i in self._bds else c
            for i, c in enumerate(chunk_coords[self._ndim_new :])
        )
        return super()._chunk_key(chunk_coords)
