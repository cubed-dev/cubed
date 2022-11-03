def take(x, indices, /, *, axis):
    return x[(slice(None),) * axis + (indices,)]
