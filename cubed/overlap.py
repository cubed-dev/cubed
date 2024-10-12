from typing import Tuple

from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import map_selection
from cubed.types import T_RectangularChunks
from cubed.utils import _cumsum
from cubed.vendor.dask.array.core import normalize_chunks
from cubed.vendor.dask.array.overlap import coerce_boundary, coerce_depth
from cubed.vendor.dask.utils import has_keyword


def map_overlap(
    func,
    *args,
    dtype=None,
    chunks=None,
    depth=None,
    boundary=None,
    trim=False,
    **kwargs,
):
    """Apply a function to corresponding blocks from multiple input arrays with some overlap.

    Parameters
    ----------
    func : callable
        Function to apply to every block (with overlap) to produce the output array.
    args : arrays
        The Cubed arrays to map over. Note that currently only one array may be specified.
    dtype : np.dtype
        The ``dtype`` of the output array.
    chunks : tuple
        Chunk shape of blocks in the output array.
    depth :  int, tuple, dict or list
        The number of elements that each block should share with its neighbors.
    boundary : value type, tuple, dict or list
        How to handle the boundaries. Note that this currently only supports constant values.
    trim : bool
        Whether or not to trim ``depth`` elements from each block after calling the map function.
        Currently only ``False`` is supported.
    **kwargs : dict
        Extra keyword arguments to pass to function.
    """
    if trim:
        raise ValueError("trim is not supported")

    chunks = normalize_chunks(chunks, dtype=dtype)
    shape = tuple(map(sum, chunks))

    # Coerce depth and boundary arguments to lists of individual
    # specifications for each array argument
    def coerce(xs, arg, fn):
        if not isinstance(arg, list):
            arg = [arg] * len(xs)
        return [fn(x.ndim, a) for x, a in zip(xs, arg)]

    depth = coerce(args, depth, coerce_depth)
    boundary = coerce(args, boundary, coerce_boundary)

    x = args[0]  # TODO: support multiple input arrays

    def selection_function(out_key):
        out_coords = out_key[1:]
        block_id = out_coords
        return get_item_with_depth(x.chunks, block_id, depth[0])

    max_num_input_blocks = _overlap_num_input_blocks(depth[0], x.numblocks)

    has_block_id_kw = has_keyword(func, "block_id")

    # First read the chunk with overlaps determined by depth, then pad boundaries second.
    # Do it this way round so we can do everything with one blockwise. The alternative,
    # which pads the entire array first (via concatenate), would result in at least one extra copy.

    return map_selection(
        _overlap,
        selection_function,
        x,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        max_num_input_blocks=max_num_input_blocks,
        overlap_func=func,
        depth=depth,
        boundary=boundary,
        numblocks=x.numblocks,
        has_block_id_kw=has_block_id_kw,
    )


def _overlap_num_input_blocks(depth, numblocks):
    num = 1
    for i in depth.keys():
        num *= min(numblocks[i], 3)
    return num


def _overlap(
    a,
    overlap_func=None,
    depth=None,
    boundary=None,
    numblocks=None,
    has_block_id_kw=False,
    block_id=None,
    **kwargs,
):
    depth = depth[0]
    boundary = boundary[0]
    out = _pad_boundaries(a, depth, boundary, numblocks, block_id)
    if has_block_id_kw:
        return overlap_func(out, block_id=block_id, **kwargs)
    else:
        return overlap_func(out, **kwargs)


def _clamp(minimum: int, x: int, maximum: int) -> int:
    return max(minimum, min(x, maximum))


def get_item_with_depth(
    chunks: T_RectangularChunks, idx: Tuple[int, ...], depth
) -> Tuple[slice, ...]:
    """Convert a chunk index to a tuple of slices with depth offsets."""
    starts = tuple(_cumsum(c, initial_zero=True) for c in chunks)
    loc = tuple(
        (
            _clamp(0, start[i] - depth[ax], start[-1]),
            _clamp(0, start[i + 1] + depth[ax], start[-1]),
        )
        for ax, (i, start) in enumerate(zip(idx, starts))
    )
    return tuple(slice(*s, None) for s in loc)


def _pad_boundaries(x, depth, boundary, numblocks, block_id):
    for i in range(x.ndim):
        d = depth.get(i, 0)
        if d == 0 or block_id[i] not in (0, numblocks[i] - 1):
            continue
        pad_shape = list(x.shape)
        pad_shape[i] = d
        pad_shape = tuple(pad_shape)
        p = nxp.full_like(x, fill_value=boundary[i], shape=pad_shape)
        if block_id[i] == 0:  # first block on axis i
            x = nxp.concat([p, x], axis=i)
        if block_id[i] == numblocks[i] - 1:  # last block on axis i
            x = nxp.concat([x, p], axis=i)
    return x
