from cubed.array_api.manipulation_functions import concat


def pad(x, pad_width, mode=None, chunks=None):
    """Pad an array."""
    if len(pad_width) != x.ndim:
        raise ValueError("`pad_width` must have as many entries as array dimensions")
    axis = tuple(
        i
        for (i, (before, after)) in enumerate(pad_width)
        if (before != 0 or after != 0)
    )
    if len(axis) != 1:
        raise ValueError("only one axis can be padded")
    axis = axis[0]
    if pad_width[axis] != (1, 0):
        raise ValueError("only a pad width of (1, 0) is allowed")
    if mode != "symmetric":
        raise ValueError(f"Mode is not supported: {mode}")

    select = []
    for i in range(x.ndim):
        if i == axis:
            select.append(slice(0, 1))
        else:
            select.append(slice(None))
    select = tuple(select)
    a = x[select]
    return concat([a, x], axis=axis, chunks=chunks or x.chunksize)
