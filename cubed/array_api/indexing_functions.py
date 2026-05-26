def take(x, indices, /, *, axis=None):
    from cubed.array_api.manipulation_functions import flatten
    from cubed.vendor.dask.array.utils import validate_axis

    if axis is None:
        x = flatten(x)
        axis = 0
    else:
        axis = validate_axis(axis, x.ndim)
    return x[(slice(None),) * axis + (indices,)]
