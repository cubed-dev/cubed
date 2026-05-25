import numpy as np


def sliding_window_view(x, window_shape, axis=None, **kwargs):
    """Create a sliding window view of an array.

    Uses rechunk + map_blocks so each rolling axis is a single chunk.
    New window dimensions are appended at the end of the output shape.
    """
    from cubed.core.ops import map_blocks, rechunk

    # Normalize to tuples
    if axis is None:
        axis = tuple(range(x.ndim))
        if isinstance(window_shape, int):
            window_shape = (window_shape,) * x.ndim
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(window_shape, int):
        window_shape = (window_shape,)

    # Rechunk rolling axes to one chunk each (required for correctness)
    rechunk_spec = {ax: x.shape[ax] for ax in axis}
    x_r = rechunk(x, rechunk_spec)

    # Build output chunks: rolling axes shrink by W-1, window dims appended
    out_chunks = list(x_r.chunks)
    for ax, W in zip(axis, window_shape):
        out_chunks[ax] = tuple(s - W + 1 for s in x_r.chunks[ax])
    for W in window_shape:
        out_chunks.append((W,))

    new_axis = list(range(x.ndim, x.ndim + len(axis)))

    return map_blocks(
        _sliding_window_view_block,
        x_r,
        dtype=x.dtype,
        chunks=tuple(out_chunks),
        new_axis=new_axis,
        window_shape=window_shape,
        axis=axis,
    )


def _sliding_window_view_block(block, window_shape, axis, **kwargs):
    return np.lib.stride_tricks.sliding_window_view(
        block, window_shape=window_shape, axis=axis
    )
