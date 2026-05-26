from cubed.array_api.creation_functions import full
from cubed.array_api.manipulation_functions import concat


def pad(x, pad_width, mode=None, constant_values=0, chunks=None):
    """Pad an array."""
    if len(pad_width) != x.ndim:
        raise ValueError("`pad_width` must have as many entries as array dimensions")

    if mode == "constant":
        return _pad_constant(x, pad_width, constant_values, chunks)
    elif mode == "symmetric":
        return _pad_symmetric(x, pad_width, chunks)
    else:
        raise ValueError(f"Mode is not supported: {mode}")


def _pad_constant(x, pad_width, constant_values, chunks):
    cv = _normalize_constant_values(constant_values, x.ndim)
    result = x
    for axis, ((pad_before, pad_after), (val_before, val_after)) in enumerate(
        zip(pad_width, cv)
    ):
        if pad_before == 0 and pad_after == 0:
            continue
        arrays = []
        if pad_before > 0:
            shape = list(result.shape)
            shape[axis] = pad_before
            c = list(result.chunksize)
            c[axis] = min(pad_before, result.chunksize[axis])
            arrays.append(
                full(
                    tuple(shape),
                    val_before,
                    dtype=result.dtype,
                    chunks=tuple(c),
                    spec=result.spec,
                )
            )
        arrays.append(result)
        if pad_after > 0:
            shape = list(result.shape)
            shape[axis] = pad_after
            c = list(result.chunksize)
            c[axis] = min(pad_after, result.chunksize[axis])
            arrays.append(
                full(
                    tuple(shape),
                    val_after,
                    dtype=result.dtype,
                    chunks=tuple(c),
                    spec=result.spec,
                )
            )
        result = concat(arrays, axis=axis, chunks=chunks or x.chunksize)
    return result


def _normalize_constant_values(constant_values, ndim):
    """Normalize constant_values to a list of (before, after) per axis.

    Accepts a scalar, a (before, after) pair, or a sequence of ndim pairs.
    """
    try:
        iter(constant_values)
    except TypeError:
        # scalar
        return [(constant_values, constant_values)] * ndim

    cv = list(constant_values)
    if len(cv) == 2 and not hasattr(cv[0], "__len__"):
        # (before, after) pair applied to every axis
        return [(cv[0], cv[1])] * ndim
    if len(cv) == ndim:
        # per-axis sequence of (before, after) pairs
        return [(pair[0], pair[1]) for pair in cv]
    raise ValueError(f"Invalid constant_values for ndim={ndim}: {constant_values}")


def _pad_symmetric(x, pad_width, chunks):
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

    select = tuple(slice(0, 1) if i == axis else slice(None) for i in range(x.ndim))
    a = x[select]
    return concat([a, x], axis=axis, chunks=chunks or x.chunksize)
