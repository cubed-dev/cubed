from bisect import bisect
from operator import add, mul

import numpy as np
import tlz
from toolz import reduce

from cubed.array_api.creation_functions import empty
from cubed.backend_array_api import namespace as nxp
from cubed.core import squeeze  # noqa: F401
from cubed.core import blockwise, rechunk, unify_chunks
from cubed.core.ops import elemwise, general_blockwise, map_blocks, map_direct
from cubed.utils import block_id_to_offset, get_item, offset_to_block_id, to_chunksize
from cubed.vendor.dask.array.core import broadcast_chunks, normalize_chunks
from cubed.vendor.dask.array.reshape import reshape_rechunk
from cubed.vendor.dask.array.utils import validate_axis


def broadcast_arrays(*arrays):
    # From dask broadcast_arrays

    # Unify uneven chunking
    inds = [list(reversed(range(x.ndim))) for x in arrays]
    uc_args = tlz.concat(zip(arrays, inds))
    _, args = unify_chunks(*uc_args, warn=False)

    shape = np.broadcast_shapes(*(e.shape for e in args))
    chunks = broadcast_chunks(*(e.chunks for e in args))

    result = tuple(broadcast_to(e, shape=shape, chunks=chunks) for e in args)

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
    template = empty(shape, dtype=nxp.int8, chunks=chunks, spec=x.spec)

    return elemwise(_broadcast_like, x, template, dtype=x.dtype)


def _broadcast_like(x, template):
    return nxp.broadcast_to(x, template.shape)


def concat(arrays, /, *, axis=0):
    if not arrays:
        raise ValueError("Need array(s) to concat")

    if axis is None:
        arrays = [flatten(array) for array in arrays]
        axis = 0

    # TODO: check arrays all have same shape (except in the dimension specified by axis)
    # TODO: type promotion
    # TODO: unify chunks

    a = arrays[0]

    # offsets along axis for the start of each array
    offsets = [0] + list(tlz.accumulate(add, [a.shape[axis] for a in arrays]))

    axis = validate_axis(axis, a.ndim)
    shape = a.shape[:axis] + (offsets[-1],) + a.shape[axis + 1 :]
    dtype = a.dtype
    chunks = normalize_chunks(to_chunksize(a.chunks), shape=shape, dtype=dtype)

    # memory allocated by reading one chunk from input array
    # note that although the output chunk will overlap multiple input chunks,
    # the chunks are read in series, reusing memory
    extra_projected_mem = a.chunkmem

    return map_direct(
        _read_concat_chunk,
        *arrays,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        extra_projected_mem=extra_projected_mem,
        axis=axis,
        offsets=offsets,
    )


def _read_concat_chunk(x, *arrays, axis=None, offsets=None, block_id=None):
    # determine the start and stop indexes for this block along the axis dimension
    chunks = arrays[0].zarray.chunks
    start = block_id[axis] * chunks[axis]
    stop = start + x.shape[axis]

    # produce a key that has slices (except for axis dimension, which is replaced below)
    idx = tuple(0 if i == axis else v for i, v in enumerate(block_id))
    key = get_item(arrays[0].chunks, idx)

    # concatenate slices of the arrays
    parts = []
    for ai, sl in _array_slices(offsets, start, stop):
        key = tuple(sl if i == axis else k for i, k in enumerate(key))
        parts.append(arrays[ai].zarray[key])
    return nxp.concat(parts, axis=axis)


def _array_slices(offsets, start, stop):
    """Return pairs of array index and slice to slice from start to stop in the concatenated array."""
    slice_start = start
    while slice_start < stop:
        # find array that slice_start falls in
        i = bisect(offsets, slice_start) - 1
        slice_stop = min(stop, offsets[i + 1])
        yield i, slice(slice_start - offsets[i], slice_stop - offsets[i])
        slice_start = slice_stop


def expand_dims(x, /, *, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    ndim_new = len(axis) + x.ndim
    axis = validate_axis(axis, ndim_new)

    chunks_it = iter(x.chunks)
    chunks = tuple(1 if i in axis else next(chunks_it) for i in range(ndim_new))

    return map_blocks(
        _expand_dims, x, dtype=x.dtype, chunks=chunks, new_axis=axis, axis=axis
    )


def _expand_dims(a, *args, **kwargs):
    if isinstance(a, dict):
        return {k: nxp.expand_dims(v, *args, **kwargs) for k, v in a.items()}
    return nxp.expand_dims(a, *args, **kwargs)


def flatten(x):
    return reshape(x, (-1,))


def moveaxis(
    x,
    source,
    destination,
    /,
):
    # From NumPy: https://github.com/numpy/numpy/blob/a4120979d216cce00dcee511aad70bf7b45ef6e0/numpy/core/numeric.py#L1389-L1457
    from numpy.core.numeric import normalize_axis_tuple

    source = normalize_axis_tuple(source, x.ndim, "source")
    destination = normalize_axis_tuple(destination, x.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have "
            "the same number of elements"
        )

    order = [n for n in range(x.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    result = permute_dims(x, order)
    return result


def permute_dims(x, /, axes):
    # From dask transpose
    if axes:
        if len(axes) != x.ndim:
            raise ValueError("axes don't match array")
    else:
        axes = tuple(range(x.ndim))[::-1]
    axes = tuple(d + x.ndim if d < 0 else d for d in axes)

    # extra memory copy due to Zarr enforcing C order on transposed array
    extra_projected_mem = x.chunkmem
    return blockwise(
        nxp.permute_dims,
        axes,
        x,
        tuple(range(x.ndim)),
        dtype=x.dtype,
        axes=axes,
        extra_projected_mem=extra_projected_mem,
    )


def reshape(x, /, shape, *, copy=None):
    # based on dask reshape

    known_sizes = [s for s in shape if s != -1]
    if len(known_sizes) != len(shape):
        if len(shape) - len(known_sizes) > 1:
            raise ValueError("can only specify one unknown dimension")
        # Fastpath for x.reshape(-1) on 1D arrays
        if len(shape) == 1 and x.ndim == 1:
            return x
        missing_size = x.size // reduce(mul, known_sizes, 1)
        shape = tuple(missing_size if s == -1 else s for s in shape)

    if reduce(mul, shape, 1) != x.size:
        raise ValueError("total size of new array must be unchanged")

    if x.shape == shape:
        return x

    if x.npartitions == 1:
        outchunks = tuple((d,) for d in shape)
        return reshape_chunks(x, shape, outchunks)

    inchunks, outchunks = reshape_rechunk(x.shape, shape, x.chunks)

    # TODO: make sure chunks are not too large

    x2 = rechunk(x, to_chunksize(inchunks))

    return reshape_chunks(x2, shape, outchunks)


def reshape_chunks(x, shape, chunks):
    if reduce(mul, shape, 1) != x.size:
        raise ValueError("total size of new array must be unchanged")

    # TODO: check number of chunks is unchanged
    # inchunks = normalize_chunks(x.chunks, shape=x.shape, dtype=x.dtype)
    outchunks = normalize_chunks(chunks, shape=shape, dtype=x.dtype)

    # use an empty template (handles smaller end chunks)
    template = empty(shape, dtype=x.dtype, chunks=chunks, spec=x.spec)

    def key_function(out_key):
        out_coords = out_key[1:]
        offset = block_id_to_offset(out_coords, template.numblocks)
        in_coords = offset_to_block_id(offset, x.numblocks)
        return (
            (x.name, *in_coords),
            (template.name, *out_coords),
        )

    return general_blockwise(
        _reshape_chunk,
        key_function,
        x,
        template,
        shape=shape,
        dtype=x.dtype,
        chunks=outchunks,
    )


def _reshape_chunk(x, template):
    return nxp.reshape(x, template.shape)


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

    array_names = [a.name for a in arrays]

    def key_function(out_key):
        out_coords = out_key[1:]
        in_name = array_names[out_coords[axis]]
        return ((in_name, *(out_coords[:axis] + out_coords[(axis + 1) :])),)

    # We have to mark this as fusable=False since the number of input args to
    # the _read_stack_chunk function is *not* the same as the number of
    # predecessor nodes in the DAG, and the fusion functions in blockwise
    # assume they are the same. See https://github.com/cubed-dev/cubed/issues/414
    return general_blockwise(
        _read_stack_chunk,
        key_function,
        *arrays,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        axis=axis,
        fusable=False,
    )


def _read_stack_chunk(array, axis=None):
    return nxp.expand_dims(array, axis=axis)
