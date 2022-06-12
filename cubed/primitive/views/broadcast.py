import numpy as np

from .zarr_view import Translator, ZarrArrayView


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

    return ZarrArrayView(source, BroadcastViewTranslator(shape, chunks, bds, ndim_new))


class BroadcastViewTranslator(Translator):
    def __init__(self, shape, chunks, bds, ndim_new):
        self.shape = shape
        self.chunks = chunks
        self.bds = bds
        self.ndim_new = ndim_new

    def to_source_chunk_coords(self, target_chunk_coords):
        # change chunk_coords to original (pre-broadcast) values
        return tuple(
            0 if i in self.bds else c
            for i, c in enumerate(target_chunk_coords[self.ndim_new :])
        )

    def to_target_chunk(self, source_chunk):
        return np.broadcast_to(source_chunk, shape=self.chunks)
