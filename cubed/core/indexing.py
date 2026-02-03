import math
from operator import add
from typing import TYPE_CHECKING

import ndindex
import numpy as np
from toolz import accumulate, map

from cubed.backend_array_api import backend_array_to_numpy_array
from cubed.core.array import CoreArray
from cubed.core.ops import general_blockwise, map_selection, merge_chunks
from cubed.primitive.blockwise import ChunkKey, FunctionArgs, KeyFunctionResult
from cubed.utils import array_size, normalize_chunks

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array


def index(x, key):
    "Subset an array, along one or more axes."
    if not isinstance(key, tuple):
        key = (key,)

    # Replace Cubed arrays with NumPy arrays - note that this may trigger a computation!
    # Note that NumPy arrays are needed for ndindex.
    key = tuple(
        backend_array_to_numpy_array(dim_sel.compute())
        if isinstance(dim_sel, CoreArray)
        else dim_sel
        for dim_sel in key
    )

    # Canonicalize index
    idx = ndindex.ndindex(key)
    idx = idx.expand(x.shape)

    # Remove newaxis values, to be filled in with expand_dims at end
    where_newaxis = [
        i for i, ia in enumerate(idx.args) if isinstance(ia, ndindex.Newaxis)
    ]
    for i, a in enumerate(where_newaxis):
        n = sum(isinstance(ia, ndindex.Integer) for ia in idx.args[:a])
        if n:
            where_newaxis[i] -= n
    idx = ndindex.Tuple(*(ia for ia in idx.args if not isinstance(ia, ndindex.Newaxis)))
    selection = idx.raw

    # Check selection is supported
    if any(ia.step < 1 for ia in idx.args if isinstance(ia, ndindex.Slice)):
        raise NotImplementedError(f"Slice step must be >= 1: {key}")
    if not all(
        isinstance(ia, (ndindex.Integer, ndindex.Slice, ndindex.IntegerArray))
        for ia in idx.args
    ):
        raise NotImplementedError(
            "Only integer, slice, or integer array indexes are allowed."
        )
    if sum(1 for ia in idx.args if isinstance(ia, ndindex.IntegerArray)) > 1:
        raise NotImplementedError("Only one integer array index is allowed.")

    # Use ndindex to find the resulting array shape and chunks

    def chunk_len_for_indexer(ia, c):
        if not isinstance(ia, ndindex.Slice):
            return c
        return max(c // ia.step, 1)

    def merged_chunk_len_for_indexer(ia, c):
        if not isinstance(ia, ndindex.Slice):
            return c
        if ia.step == 1:
            return c
        if (c // ia.step) < 1:
            return c
        # note that this may not be the same as c
        # but it is guaranteed to be a multiple of the corresponding
        # value returned by chunk_len_for_indexer, which is required
        # by merge_chunks
        return (c // ia.step) * ia.step

    shape = idx.newshape(x.shape)

    if shape == x.shape:
        # no op case (except possibly newaxis applied below)
        out = x
    elif array_size(shape) == 0:
        # empty output case
        from cubed.array_api.creation_functions import empty

        chunks = tuple(c for c in x.chunksize if c > 0)
        out = empty(shape, dtype=x.dtype, chunks=chunks, spec=x.spec)
    else:
        dtype = x.dtype
        chunks = tuple(
            chunk_len_for_indexer(ia, c)
            for ia, c in zip(idx.args, x.chunksize)
            if not isinstance(ia, ndindex.Integer)
        )

        # this is the same as chunks, except it has the same number of dimensions as the input
        out_chunksizes = tuple(
            chunk_len_for_indexer(ia, c) if not isinstance(ia, ndindex.Integer) else 1
            for ia, c in zip(idx.args, x.chunksize)
        )

        target_chunks = normalize_chunks(chunks, shape, dtype=dtype)

        # use map_selection (which uses general_blockwise) to allow more opportunities for optimization than map_direct

        def selection_function(out_key):
            out_coords = out_key.coords
            return _target_chunk_selection(target_chunks, out_coords, selection)

        max_num_input_blocks = _index_num_input_blocks(
            idx, x.chunksize, out_chunksizes, x.numblocks
        )

        out = map_selection(
            None,  # no function to apply after selection
            selection_function,
            x,
            shape,
            x.dtype,
            target_chunks,
            max_num_input_blocks=max_num_input_blocks,
        )

        # merge chunks for any dims with step > 1 so they are
        # the same size as the input (or slightly smaller due to rounding)
        merged_chunks = tuple(
            merged_chunk_len_for_indexer(ia, c)
            for ia, c in zip(idx.args, x.chunksize)
            if not isinstance(ia, ndindex.Integer)
        )
        if chunks != merged_chunks:
            out = merge_chunks(out, merged_chunks)

    for axis in where_newaxis:
        from cubed.array_api.manipulation_functions import expand_dims

        out = expand_dims(out, axis=axis)

    return out


def _index_num_input_blocks(
    idx: ndindex.Tuple, in_chunksizes, out_chunksizes, numblocks
):
    num = 1
    for ia, c, oc, nb in zip(idx.args, in_chunksizes, out_chunksizes, numblocks):
        if isinstance(ia, ndindex.Integer) or nb == 1:
            pass  # single block
        elif isinstance(ia, ndindex.Slice):
            if (ia.start // c) == ((ia.stop - 1) // c):
                pass  # within same block
            elif ia.start % c != 0:
                num *= 2  # doesn't start on chunk boundary
            elif ia.step is not None and c % ia.step != 0 and oc > 1:
                # step is not a multiple of chunk size, and output chunks have more than one element
                # so some output chunks will access two input chunks
                num *= 2
        elif isinstance(ia, ndindex.IntegerArray):
            # in the worse case, elements could be retrieved from all blocks
            # TODO: improve to calculate the actual max input blocks
            num *= nb
        else:
            raise NotImplementedError(
                "Only integer, slice, or int array indexes are supported."
            )
    return num


def _target_chunk_selection(target_chunks, idx, selection):
    # integer, integer array, and slice indexes can be interspersed in selection
    # idx is the chunk index for the output (target_chunks)

    sel = []
    i = 0  # index into target_chunks and idx
    for s in selection:
        if isinstance(s, slice):
            offset = s.start or 0
            step = s.step if s.step is not None else 1
            start = tuple(
                accumulate(add, tuple(x * step for x in target_chunks[i]), offset)
            )
            j = idx[i]
            sel.append(slice(start[j], start[j + 1], step))
            i += 1
        # ndindex uses np.ndarray for integer arrays
        elif isinstance(s, np.ndarray):
            # find the cumulative chunk starts
            target_chunk_starts = [0] + list(
                accumulate(add, [c for c in target_chunks[i]])
            )
            # and use to slice the integer array
            j = idx[i]
            sel.append(s[target_chunk_starts[j] : target_chunk_starts[j + 1]])
            i += 1
        elif isinstance(s, int):
            sel.append(s)
            # don't increment i since integer indexes don't have a dimension in the target
        else:
            raise ValueError(f"Unsupported selection: {s}")
    return tuple(sel)


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

        def key_function(out_key: ChunkKey) -> KeyFunctionResult:
            out_coords = out_key.coords
            in_coords = tuple(
                get_dim_index(ia, bi) for ia, bi in zip(idx.args, out_coords)
            )
            return FunctionArgs(ChunkKey(self.array.name, in_coords))

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
