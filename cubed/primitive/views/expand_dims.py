import numpy as np
from dask.array.utils import validate_axis

from .zarr_view import Translator, ZarrArrayView


def expand_dims(source, axis):
    """Expands the shape of an array by inserting new dimensions of size one.

    Parameters
    ----------
    source : Zarr array
    axis : int or tuple
        The axis (or axes) to insert.
    Returns
    -------
    target:  Array for the Zarr array output
    """

    if not isinstance(axis, tuple):
        axis = (axis,)

    ndim_new = len(axis) + source.ndim
    axis = validate_axis(axis, ndim_new)

    shape_it = iter(source.shape)
    shape = tuple(1 if i in axis else next(shape_it) for i in range(ndim_new))

    chunks_it = iter(source.chunks)
    chunks = tuple(1 if i in axis else next(chunks_it) for i in range(ndim_new))

    return ZarrArrayView(source, ExpandDimsViewTranslator(shape, chunks, axis))


class ExpandDimsViewTranslator(Translator):
    def __init__(self, shape, chunks, axis):
        self.shape = shape
        self.chunks = chunks
        self.axis = axis

    def to_source_chunk_coords(self, target_chunk_coords):
        return tuple(c for i, c in enumerate(target_chunk_coords) if i not in self.axis)

    def to_target_chunk(self, source_chunk):
        return np.expand_dims(source_chunk, axis=self.axis)
