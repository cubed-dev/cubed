from bisect import bisect
from operator import add, mul

import numpy as np
import tlz
from toolz import reduce

from cubed.array_api.creation_functions import empty
from cubed.backend_array_api import namespace as nxp
from cubed.core import squeeze  # noqa: F401
from cubed.core import blockwise, rechunk, unify_chunks
from cubed.core.ops import (
    _create_zarr_indexer,
    elemwise,
    general_blockwise,
    map_blocks,
    map_selection,
)
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

    if chunks is None:
        # New dimensions and broadcast dimensions have chunk size 1
        # This behaviour differs from dask where it is the full dimension size
        xchunks = normalize_chunks(x.chunks, x.shape, dtype=x.dtype)

        def chunklen(shapelen):
            return (1,) * shapelen if shapelen > 0 else (0,)

        chunks = tuple(chunklen(s) for s in shape[:ndim_new]) + tuple(
            bd if old > 1 else chunklen(new)
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


def concat(arrays, /, *, axis=0, chunks=None):
    if not arrays:
        raise ValueError("Need array(s) to concat")

    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("concat inputs must all have the same dtype")

    if axis is None:
        arrays = [flatten(array) for array in arrays]
        axis = 0

    if len(arrays) == 1:
        return arrays[0]

    a = arrays[0]

    # check arrays all have same shape (except in the dimension specified by axis)
    ndim = a.ndim
    if not all(
        i == axis or all(x.shape[i] == arrays[0].shape[i] for x in arrays)
        for i in range(ndim)
    ):
        raise ValueError(
            f"all the input array dimensions except for the concatenation axis must match exactly: {[x.shape for x in arrays]}"
        )

    # check arrays all have the same chunk size along axis (if more than one chunk)
    if len({a.chunksize[axis] for a in arrays if a.numblocks[axis] > 1}) > 1:
        raise ValueError(
            f"all the input array chunk sizes must match along the concatenation axis: {[x.chunksize[axis] for x in arrays]}"
        )

    # unify chunks (except in the dimension specified by axis)
    inds = [list(range(x.ndim)) for x in arrays]
    for i, ind in enumerate(inds):
        ind[axis] = -(i + 1)
    uc_args = tlz.concat(zip(arrays, inds))
    chunkss, arrays = unify_chunks(*uc_args, warn=False)

    # offsets along axis for the start of each array
    offsets = [0] + list(tlz.accumulate(add, [a.shape[axis] for a in arrays]))
    in_shapes = tuple(array.shape for array in arrays)

    axis = validate_axis(axis, ndim)
    shape = a.shape[:axis] + (offsets[-1],) + a.shape[axis + 1 :]
    dtype = a.dtype
    if chunks is None:
        # use unified chunks except for dimension specified by axis
        axis_chunksize = max(a.chunksize[axis] for a in arrays)
        chunksize = tuple(
            axis_chunksize if i == axis else chunkss[i] for i in range(ndim)
        )
        chunks = normalize_chunks(chunksize, shape=shape, dtype=dtype)
    else:
        chunks = normalize_chunks(chunks, shape=shape, dtype=dtype)

    def key_function(out_key):
        out_coords = out_key[1:]
        block_id = out_coords

        # determine the start and stop indexes for this block along the axis dimension
        chunksize = to_chunksize(chunks)
        start = block_id[axis] * chunksize[axis]
        stop = start + chunksize[axis]
        stop = min(stop, shape[axis])

        # produce a key that has slices (except for axis dimension, which is replaced below)
        idx = tuple(0 if i == axis else v for i, v in enumerate(block_id))
        key = get_item(chunks, idx)

        # find slices of the arrays
        in_keys = []
        for ai, sl in _array_slices(offsets, start, stop):
            key = tuple(sl if i == axis else k for i, k in enumerate(key))

            # use a Zarr BasicIndexer to convert this to input coordinates
            a = arrays[ai]
            indexer = _create_zarr_indexer(key, a.shape, a.chunksize)

            in_keys.extend([(a.name,) + cp.chunk_coords for cp in indexer])

        return (iter(tuple(in_key for in_key in in_keys)),)

    num_input_blocks = (1,) * len(arrays)
    iterable_input_blocks = (True,) * len(arrays)

    # We have to mark this as fusable_with_predecessors=False since the number of input args to
    # the _read_concat_chunk function is *not* the same as the number of
    # predecessor nodes in the DAG, and the fusion functions in blockwise
    # assume they are the same. See https://github.com/cubed-dev/cubed/issues/414
    # This also affects stack.
    return general_blockwise(
        _read_concat_chunk,
        key_function,
        *arrays,
        shapes=[shape],
        dtypes=[dtype],
        chunkss=[chunks],
        num_input_blocks=num_input_blocks,
        iterable_input_blocks=iterable_input_blocks,
        extra_func_kwargs=dict(dtype=dtype),
        target_shape=shape,
        target_chunks=chunks,
        axis=axis,
        offsets=offsets,
        in_shapes=in_shapes,
        function_nargs=1,
        fusable_with_predecessors=False,
    )


def _read_concat_chunk(
    arrays,
    dtype=None,
    target_shape=None,
    target_chunks=None,
    axis=None,
    offsets=None,
    in_shapes=None,
    block_id=None,
):
    # determine the start and stop indexes for this block along the axis dimension
    chunksize = to_chunksize(target_chunks)
    start = block_id[axis] * chunksize[axis]
    stop = start + chunksize[axis]
    stop = min(stop, target_shape[axis])

    chunk_shape = tuple(ch[bi] for ch, bi in zip(target_chunks, block_id))
    out = np.empty(chunk_shape, dtype=dtype)
    for array, (lchunk_selection, lout_selection) in zip(
        arrays,
        _chunk_slices(
            offsets, start, stop, target_chunks, chunksize, in_shapes, axis, block_id
        ),
    ):
        out[lout_selection] = array[lchunk_selection]
    return out


def _array_slices(offsets, start, stop):
    """Return pairs of array index and array slice to slice from start to stop in the concatenated array."""
    slice_start = start
    while slice_start < stop:
        # find array that slice_start falls in
        i = bisect(offsets, slice_start) - 1
        slice_stop = min(stop, offsets[i + 1])
        yield i, slice(slice_start - offsets[i], slice_stop - offsets[i])
        slice_start = slice_stop


def _chunk_slices(
    offsets, start, stop, target_chunks, chunksize, in_shapes, axis, block_id
):
    """Return pairs of chunk slices to slice input array chunks and output concatenated chunk."""

    # an output chunk may have selections from more than one array, so we need an offset per array
    arr_sel_offset = 0  # offset along axis

    # produce a key that has slices (except for axis dimension, which is replaced below)
    idx = tuple(0 if i == axis else v for i, v in enumerate(block_id))
    key = get_item(target_chunks, idx)

    for ai, sl in _array_slices(offsets, start, stop):
        key = tuple(sl if i == axis else k for i, k in enumerate(key))
        indexer = _create_zarr_indexer(key, in_shapes[ai], chunksize)
        for cp in indexer:
            lout_selection_with_offset = tuple(
                sl
                if ax != axis
                else slice(sl.start + arr_sel_offset, sl.stop + arr_sel_offset)
                for ax, sl in enumerate(cp.out_selection)
            )
            yield cp.chunk_selection, lout_selection_with_offset

        arr_sel_offset += cp.out_selection[axis].stop


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


def flip(x, /, *, axis=None):
    if axis is None:
        axis = tuple(range(x.ndim))  # all axes
    if not isinstance(axis, tuple):
        axis = (axis,)
    axis = validate_axis(axis, x.ndim)

    def selection_function(out_key):
        out_coords = out_key[1:]
        block_id = out_coords

        # produce a key that has slices (except for axis dimensions, which are replaced below)
        idx = tuple(0 if i == axis else v for i, v in enumerate(block_id))
        key = list(get_item(x.chunks, idx))

        for ax in axis:
            # determine the start and stop indexes for this block along the axis dimension
            start = block_id[ax] * x.chunksize[ax]
            stop = start + x.chunksize[ax]
            stop = min(stop, x.shape[ax])

            # flip start and stop
            axis_len = x.shape[ax]
            start, stop = axis_len - stop, axis_len - start

            # replace with slice
            key[ax] = slice(start, stop)

        return tuple(key)

    max_num_input_blocks = _flip_num_input_blocks(axis, x.shape, x.chunksize)

    return map_selection(
        nxp.flip,
        selection_function,
        x,
        shape=x.shape,
        dtype=x.dtype,
        chunks=x.chunks,
        max_num_input_blocks=max_num_input_blocks,
        axis=axis,
    )


def _flip_num_input_blocks(axis, shape, chunksizes):
    num = 1
    for ax in axis:
        if shape[ax] % chunksizes[ax] != 0:
            num *= 2
    return num


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


def repeat(x, repeats, /, *, axis=0):
    if not isinstance(repeats, int):
        raise ValueError("repeat only supports integral values for `repeats`")

    if axis is None:
        x = flatten(x)
        axis = 0

    shape = x.shape[:axis] + (x.shape[axis] * repeats,) + x.shape[axis + 1 :]
    chunks = normalize_chunks(x.chunksize, shape=shape, dtype=x.dtype)

    # This implementation calls nxp.repeat in every output block, which is 'repeats' times
    # more than necessary than if we had a primitive op that could write multiple blocks.

    def key_function(out_key):
        out_coords = out_key[1:]
        in_coords = tuple(
            bi // repeats if i == axis else bi for i, bi in enumerate(out_coords)
        )
        return ((x.name, *in_coords),)

    # extra memory from calling 'nxp.repeat' on a chunk
    extra_projected_mem = x.chunkmem * repeats
    return general_blockwise(
        _repeat,
        key_function,
        x,
        shapes=[shape],
        dtypes=[x.dtype],
        chunkss=[chunks],
        extra_projected_mem=extra_projected_mem,
        repeats=repeats,
        axis=axis,
        chunksize=x.chunksize,
    )


def _repeat(x, repeats, axis=None, chunksize=None, block_id=None):
    out = nxp.repeat(x, repeats, axis=axis)
    bi = block_id[axis] % repeats
    ind = tuple(
        slice(bi * chunksize[i], (bi + 1) * chunksize[i]) if i == axis else slice(None)
        for i in range(x.ndim)
    )
    return out[ind]


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
        shapes=[shape],
        dtypes=[x.dtype],
        chunkss=[outchunks],
    )


def _reshape_chunk(x, template):
    return nxp.reshape(x, template.shape)


def roll(x, /, shift, *, axis=None):
    # based on dask roll
    result = x

    if axis is None:
        result = flatten(result)

        if not isinstance(shift, int):
            raise TypeError("Expect `shift` to be an int when `axis` is None.")

        shift = (shift,)
        axis = (0,)
    else:
        if not isinstance(shift, tuple):
            shift = (shift,)
        if not isinstance(axis, tuple):
            axis = (axis,)

    if len(shift) != len(axis):
        raise ValueError("Must have the same number of shifts as axes.")

    for i, s in zip(axis, shift):
        shape = result.shape[i]
        s = 0 if shape == 0 else -s % shape

        sl1 = result.ndim * [slice(None)]
        sl2 = result.ndim * [slice(None)]

        sl1[i] = slice(s, None)
        sl2[i] = slice(None, s)

        sl1 = tuple(sl1)
        sl2 = tuple(sl2)

        # note we want the concatenated array to have the same chunking as input,
        # not the chunking of result[sl1], which may be different
        result = concat([result[sl1], result[sl2]], axis=i, chunks=result.chunks)

    return reshape(result, x.shape)


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

    # We have to mark this as fusable_with_predecessors=False since the number of input args to
    # the _read_stack_chunk function is *not* the same as the number of
    # predecessor nodes in the DAG, and the fusion functions in blockwise
    # assume they are the same. See https://github.com/cubed-dev/cubed/issues/414
    return general_blockwise(
        _read_stack_chunk,
        key_function,
        *arrays,
        shapes=[shape],
        dtypes=[dtype],
        chunkss=[chunks],
        axis=axis,
        function_nargs=1,
        fusable_with_predecessors=False,
    )


def _read_stack_chunk(array, axis=None):
    return nxp.expand_dims(array, axis=axis)


def tile(x, repetitions, /):
    N = len(x.shape)
    M = len(repetitions)
    if N > M:
        repetitions = (1,) * (N - M) + repetitions
    elif N < M:
        for _ in range(M - N):
            x = expand_dims(x, axis=0)
    out = x
    for i, nrep in enumerate(repetitions):
        if nrep > 1:
            out = concat([out] * nrep, axis=i)
    return out


def unstack(x, /, *, axis=0):
    axis = validate_axis(axis, x.ndim)

    n_arrays = x.shape[axis]

    if n_arrays == 0:
        return ()
    elif n_arrays == 1:
        return (squeeze(x, axis=axis),)

    shape = x.shape[:axis] + x.shape[axis + 1 :]
    dtype = x.dtype
    chunks = x.chunks[:axis] + x.chunks[axis + 1 :]

    def key_function(out_key):
        out_coords = out_key[1:]
        all_in_coords = tuple(
            out_coords[:axis] + (i,) + out_coords[axis:]
            for i in range(x.numblocks[axis])
        )
        return tuple((x.name,) + in_coords for in_coords in all_in_coords)

    return general_blockwise(
        _unstack_chunk,
        key_function,
        x,
        shapes=[shape] * n_arrays,
        dtypes=[dtype] * n_arrays,
        chunkss=[chunks] * n_arrays,
        target_stores=[None] * n_arrays,  # filled in by general_blockwise
        axis=axis,
    )


def _unstack_chunk(*arrs, axis=0):
    # unstack each array in arrs and yield all in turn
    for arr in arrs:
        for a in nxp.unstack(arr, axis=axis):
            yield a
