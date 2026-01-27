import math
from functools import reduce
from itertools import starmap
from operator import mul

from cubed.types import T_DType, T_RegularChunks, T_Shape
from cubed.utils import itemsize, normalize_dtype


class ArrayMetadata:
    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
    ):
        self.shape = shape
        self.dtype = normalize_dtype(dtype)
        self.chunks = chunks

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self) -> int:
        """Number of bytes in array"""
        return self.size * itemsize(self.dtype)

    @property
    def _cdata_shape(self) -> T_Shape:
        """The shape of the chunk grid for this array."""
        return tuple(
            starmap(
                lambda s, c: math.ceil(s / c),
                zip(self.shape, self.chunks, strict=False),
            )
        )

    @property
    def nchunks(self) -> int:
        """Number of chunks in array"""

        # TODO: need a better way of knowing the chunk grid
        if (len(self.chunks) > 0) and not isinstance(self.chunks[0], int):
            import zarr

            return zarr.RectilinearChunks(self.chunks).total_chunks
        return reduce(mul, self._cdata_shape, 1)
