import numpy as np
from dask.array.utils import validate_axis

from .zarr_view import Translator, ZarrArrayView


def squeeze(source, axis):
    """Removes singleton dimensions from an array.

    Parameters
    ----------
    source : Zarr array
    axis : int or tuple
        The axis (or axes) to squeeze.
    Returns
    -------
    target:  Array for the Zarr array output
    """

    if not isinstance(axis, tuple):
        axis = (axis,)

    if any(source.shape[i] != 1 for i in axis):
        raise ValueError("cannot squeeze axis with size other than one")

    axis = validate_axis(axis, source.ndim)

    shape = tuple(c for i, c in enumerate(source.shape) if i not in axis)
    chunks = tuple(c for i, c in enumerate(source.chunks) if i not in axis)

    return ZarrArrayView(source, SqueezeViewTranslator(shape, chunks, axis))


class SqueezeViewTranslator(Translator):
    def __init__(self, shape, chunks, axis):
        self.shape = shape
        self.chunks = chunks
        self.axis = axis
        self.ndim_old = len(shape) + len(axis)

    def to_source_chunk_coords(self, target_chunk_coords):
        coords_it = iter(target_chunk_coords)
        return tuple(
            0 if i in self.axis else next(coords_it) for i in range(self.ndim_old)
        )

    def to_target_chunk(self, source_chunk):
        return np.squeeze(source_chunk, axis=self.axis)
