def take(x, indices, /, *, axis=None):
    from cubed.array_api.manipulation_functions import flatten

    if axis is None:
        x = flatten(x)
        axis = 0
    return x[(slice(None),) * axis + (indices,)]
