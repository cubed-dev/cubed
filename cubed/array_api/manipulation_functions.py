from math import prod
from operator import mul

import numpy as np
from dask.array.core import broadcast_chunks, broadcast_shapes
from dask.array.reshape import reshape_rechunk
from dask.array.slicing import sanitize_index
from dask.array.utils import validate_axis
from tlz import concat
from toolz import reduce

from cubed.core import squeeze  # noqa: F401
from cubed.core import Array, Plan, blockwise, gensym, rechunk, unify_chunks
from cubed.core.ops import map_blocks, map_direct
from cubed.primitive.broadcast import broadcast_to as primitive_broadcast_to
from cubed.primitive.reshape import reshape_chunks as primitive_reshape_chunks
from cubed.utils import get_item, to_chunksize


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


def expand_dims(x, /, *, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    ndim_new = len(axis) + x.ndim
    axis = validate_axis(axis, ndim_new)

    chunks_it = iter(x.chunks)
    chunks = tuple(1 if i in axis else next(chunks_it) for i in range(ndim_new))

    return map_blocks(
        np.expand_dims, x, dtype=x.dtype, chunks=chunks, new_axis=axis, axis=axis
    )


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

    if x.npartitions == 1:
        outchunks = tuple((d,) for d in shape)
        name = gensym()
        spec = x.plan.spec
        target = primitive_reshape_chunks(x.zarray, shape, outchunks)
        plan = Plan(name, "reshape", target, spec, None, None, None, x)
        return Array(name, target, plan)

    inchunks, outchunks = reshape_rechunk(x.shape, shape, x.chunks)

    # TODO: make sure chunks are not too large

    x2 = rechunk(x, to_chunksize(inchunks))

    name = gensym()
    spec = x.plan.spec
    target = primitive_reshape_chunks(x2.zarray, shape, outchunks)
    plan = Plan(name, "reshape", target, spec, None, None, None, x2)
    return Array(name, target, plan)


def stack(arrays, /, *, axis=0):
    if not arrays:
        raise ValueError("Need array(s) to stack")

    # TODO: check arrays all have same shape
    # TODO: type promotion
    # TODO: unify chunks

    a = arrays[0]

    axis = validate_axis(axis, a.ndim + 1)
    shape = a.shape[:axis] + (len(arrays),) + a.shape[axis:]
    dtype = a.dtype
    chunks = a.chunks[:axis] + ((1,) * len(arrays),) + a.chunks[axis:]

    # memory allocated by reading one chunk from an input array
    # (output is already catered for in blockwise)
    extra_required_mem = np.dtype(a.dtype).itemsize * prod(to_chunksize(a.chunks))

    return map_direct(
        _read_stack_chunk,
        *arrays,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_required_mem=extra_required_mem,
        axis=axis,
    )


def _read_stack_chunk(x, *arrays, axis=None, block_id=None):
    array = arrays[block_id[axis]]
    idx = tuple(v for i, v in enumerate(block_id) if i != axis)
    out = array.zarray[get_item(array.chunks, idx)]
    out = np.expand_dims(out, axis=axis)
    return out
