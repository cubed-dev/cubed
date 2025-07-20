import math
from functools import reduce
from itertools import starmap
from operator import mul

import numpy as np

from cubed.types import T_DType, T_RegularChunks, T_Shape


class ArrayMetadata:
    def __init__(
        self,
        shape: T_Shape,
        dtype: T_DType,
        chunks: T_RegularChunks,
    ):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.chunks = chunks

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self) -> int:
        """Number of bytes in array"""
        return self.size * self.dtype.itemsize

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
        return reduce(mul, self._cdata_shape, 1)
