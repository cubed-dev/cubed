import math
from typing import TYPE_CHECKING

import ndindex
import numpy as np
from toolz import map

from cubed.core.ops import general_blockwise

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array


class BlockView:
    """An array-like interface to the blocks of an array."""

    def __init__(self, array: "Array"):
        self.array = array

    def __getitem__(self, key) -> "Array":
        if not isinstance(key, tuple):
            key = (key,)

        # Canonicalize index
        idx = ndindex.ndindex(key)
        idx = idx.expand(self.array.numblocks)

        if any(isinstance(ia, ndindex.Newaxis) for ia in idx.args):
            raise ValueError("Slicing with xp.newaxis is not supported")

        if sum(1 for ia in idx.args if isinstance(ia, ndindex.IntegerArray)) > 1:
            raise NotImplementedError("Only one integer array index is allowed.")

        # convert Integer to Slice so we don't lose dimensions
        def convert_integer_index_to_slice(ia):
            if isinstance(ia, ndindex.Integer):
                return ndindex.Slice(ia.raw, ia.raw + 1)
            return ia

        idx = ndindex.Tuple(*(convert_integer_index_to_slice(ia) for ia in idx.args))

        chunks = tuple(
            tuple(np.array(ch)[ia].tolist())
            for ia, ch in zip(idx.raw, self.array.chunks)
        )
        shape = tuple(map(sum, chunks))

        identity = lambda a: a

        def get_dim_index(ia, i):
            if isinstance(ia, ndindex.Slice):
                step = ia.step or 1
                return ia.start + (step * i)
            elif isinstance(ia, ndindex.IntegerArray):
                return ia.raw[i]
            else:
                raise NotImplementedError(
                    "Only integer, slice, or int array indexes are supported."
                )

        def key_function(out_key):
            out_coords = out_key[1:]
            in_coords = tuple(
                get_dim_index(ia, bi) for ia, bi in zip(idx.args, out_coords)
            )
            return ((self.array.name, *in_coords),)

        out = general_blockwise(
            identity,
            key_function,
            self.array,
            shapes=[shape],
            dtypes=[self.array.dtype],
            chunkss=[chunks],
        )

        from cubed import Array

        assert isinstance(out, Array)  # single output
        return out

    @property
    def size(self) -> int:
        """
        The total number of blocks in the array.
        """
        return math.prod(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The number of blocks per axis.
        """
        return self.array.numblocks
