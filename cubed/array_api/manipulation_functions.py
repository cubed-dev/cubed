from operator import mul

import numpy as np
from dask.array.core import broadcast_chunks, broadcast_shapes
from dask.array.reshape import reshape_rechunk
from dask.array.slicing import sanitize_index
from tlz import concat
from toolz import reduce

from cubed.core import squeeze  # noqa: F401
from cubed.core import Array, Plan, blockwise, gensym, rechunk, unify_chunks
from cubed.primitive.broadcast import broadcast_to as primitive_broadcast_to
from cubed.primitive.reshape import reshape_chunks as primitive_reshape_chunks
from cubed.utils import to_chunksize


def broadcast_arrays(*arrays):
    # From dask broadcast_arrays

    # Unify uneven chunking
    inds = [list(reversed(range(x.ndim))) for x in arrays]
    uc_args = concat(zip(arrays, inds))
    _, args = unify_chunks(*uc_args, warn=False)

    shape = broadcast_shapes(*(e.shape for e in args))
    chunks = broadcast_chunks(*(e.chunks for e in args))

    result = [broadcast_to(e, shape=shape, chunks=chunks) for e in args]

    return result


def broadcast_to(x, /, shape, *, chunks=None):
    if chunks is not None:
        chunks = to_chunksize(chunks)
    name = gensym()
    spec = x.plan.spec
    target = primitive_broadcast_to(x.zarray, shape, chunks=chunks)
    plan = Plan(name, "broadcast_to", target, spec, None, None, None, x)
    return Array(name, target, plan)


def permute_dims(x, /, axes):
    # From dask transpose
    if axes:
        if len(axes) != x.ndim:
            raise ValueError("axes don't match array")
    else:
        axes = tuple(range(x.ndim))[::-1]
    axes = tuple(d + x.ndim if d < 0 else d for d in axes)
    return blockwise(
        np.transpose, axes, x, tuple(range(x.ndim)), dtype=x.dtype, axes=axes
    )


def reshape(x, /, shape):
    # based on dask reshape

    shape = tuple(map(sanitize_index, shape))
    known_sizes = [s for s in shape if s != -1]
    if len(known_sizes) != len(shape):
        raise NotImplementedError("unknown dimension not supported in reshape")

    if reduce(mul, shape, 1) != x.size:
        raise ValueError("total size of new array must be unchanged")

    if x.shape == shape:
        return x

    inchunks, outchunks = reshape_rechunk(x.shape, shape, x.chunks)

    # TODO: make sure chunks are not too large

    x2 = rechunk(x, to_chunksize(inchunks))

    name = gensym()
    spec = x.plan.spec
    target = primitive_reshape_chunks(x2.zarray, shape, outchunks)
    plan = Plan(name, "reshape", target, spec, None, None, None, x2)
    return Array(name, target, plan)
