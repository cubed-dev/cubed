from itertools import product
from math import prod
from operator import mul

import numpy as np
from dask.array.core import broadcast_chunks, normalize_chunks
from dask.array.reshape import reshape_rechunk
from dask.array.slicing import sanitize_index
from dask.array.utils import validate_axis
from tlz import concat
from toolz import reduce

from cubed.array_api.creation_functions import empty
from cubed.core import squeeze  # noqa: F401
from cubed.core import Array, Plan, blockwise, gensym, rechunk, unify_chunks
from cubed.core.ops import elemwise, map_blocks, map_direct
from cubed.primitive.reshape import reshape_chunks as primitive_reshape_chunks
from cubed.utils import get_item, to_chunksize


def broadcast_arrays(*arrays):
    # From dask broadcast_arrays

    # Unify uneven chunking
    inds = [list(reversed(range(x.ndim))) for x in arrays]
    uc_args = concat(zip(arrays, inds))
    _, args = unify_chunks(*uc_args, warn=False)

    shape = np.broadcast_shapes(*(e.shape for e in args))
    chunks = broadcast_chunks(*(e.chunks for e in args))

    result = [broadcast_to(e, shape=shape, chunks=chunks) for e in args]

    return result


def broadcast_to(x, /, shape, *, chunks=None):
    if x.shape == shape and (chunks is None or chunks == x.chunks):
        return x
    ndim_new = len(shape) - x.ndim
    if ndim_new < 0 or any(
        new != old for new, old in zip(shape[ndim_new:], x.shape) if old != 1
    ):
        raise ValueError(f"cannot broadcast shape {x.shape} to shape {shape}")

    # TODO: fix case where shape has a dimension of size zero

    if chunks is None:
        # New dimensions and broadcast dimensions have chunk size 1
        # This behaviour differs from dask where it is the full dimension size
        xchunks = normalize_chunks(x.chunks, x.shape, dtype=x.dtype)
        chunks = tuple((1,) * s for s in shape[:ndim_new]) + tuple(
            bd if old > 1 else ((1,) * new if new > 0 else (0,))
            for bd, old, new in zip(xchunks, x.shape, shape[ndim_new:])
        )
    else:
        chunks = normalize_chunks(
            chunks, shape, dtype=x.dtype, previous_chunks=x.chunks
        )
        for old_bd, new_bd in zip(x.chunks, chunks[ndim_new:]):
            if old_bd != new_bd and old_bd != (1,):
                raise ValueError(
                    "cannot broadcast chunks %s to chunks %s: "
                    "new chunks must either be along a new "
                    "dimension or a dimension of size 1" % (x.chunks, chunks)
                )

    # create an empty array as a template for blockwise to do broadcasting
    template = empty(shape, dtype=np.int8, chunks=chunks, spec=x.plan.spec)

    return elemwise(_broadcast_like, x, template, dtype=x.dtype)


def _broadcast_like(x, template):
    return np.broadcast_to(x, template.shape)


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


def reshape_chunks(x, shape, chunks):
    if reduce(mul, shape, 1) != x.size:
        raise ValueError("total size of new array must be unchanged")

    inchunks = normalize_chunks(x.chunks, shape=x.shape, dtype=x.dtype)
    outchunks = normalize_chunks(chunks, shape=shape, dtype=x.dtype)

    # TODO: check number of chunks is unchanged

    # memory allocated by reading one chunk from input array
    extra_required_mem = np.dtype(x.dtype).itemsize * prod(to_chunksize(x.chunks))

    return map_direct(
        _reshape_chunk,
        x,
        shape=shape,
        dtype=x.dtype,
        chunks=outchunks,
        extra_required_mem=extra_required_mem,
        inchunks=inchunks,
        outchunks=outchunks,
    )


def _reshape_chunk(e, x, inchunks=None, outchunks=None, block_id=None):
    in_keys = list(product(*[range(len(c)) for c in inchunks]))
    out_keys = list(product(*[range(len(c)) for c in outchunks]))
    idx = in_keys[out_keys.index(block_id)]
    out = x.zarray[get_item(x.chunks, idx)]
    return out.reshape(e.shape)


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
