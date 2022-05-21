import numpy as np
from dask.array.core import broadcast_chunks, broadcast_shapes
from tlz import concat

from cubed.core import squeeze  # noqa: F401
from cubed.core import Array, Plan, blockwise, gensym, unify_chunks
from cubed.primitive import broadcast_to as primitive_broadcast_to
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
    plan = Plan(name, "broadcast_to", target, spec)
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
