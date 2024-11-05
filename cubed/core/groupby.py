from typing import TYPE_CHECKING

from cubed.array_api.manipulation_functions import broadcast_to, expand_dims
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import map_blocks, map_selection, reduction
from cubed.utils import get_item
from cubed.vendor.dask.array.core import normalize_chunks

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array


def groupby_reduction(
    x: "Array",
    by: "Array",
    func,
    combine_func=None,
    aggregate_func=None,
    axis=None,
    intermediate_dtype=None,
    dtype=None,
    keepdims=False,
    split_every=None,
    num_groups=None,
    extra_func_kwargs=None,
) -> "Array":
    """A reduction operation that performs groupby aggregations.

    Parameters
    ----------
    x: Array
        Array being grouped along one axis.
    by: Array
        Array of non-negative integers to be used as labels with which to group
        the values in ``x`` along the reduction axis. Must be a 1D array.
    func: callable
        Function to apply to each chunk of data before reduction.
    combine_func: callable
        Function which may be applied recursively to intermediate chunks of
        data. The number of chunks that are combined in each round is
        determined by the ``split_every`` parameter. The output of the
        function is a chunk with size ``num_groups`` along the reduction axis.
    aggregate_func: callable, optional
        Function to apply to each of the final chunks to produce the final output.
    axis: int or sequence of ints, optional
        Axis to aggregate along. Only supports a single axis.
    intermediate_dtype: dtype
        Data type of intermediate output.
    dtype: dtype
        Data type of output.
    keepdims: boolean, optional
        Whether the reduction function should preserve the reduced axes,
        or remove them.
    split_every: int >= 2 or dict(axis: int), optional
        The number of chunks to combine in one round along each axis in the
        recursive aggregation.
    num_groups: int
        The number of groups in the grouping array ``by``.
    extra_func_kwargs: dict, optional
        Extra keyword arguments to pass to ``func`` and ``combine_func``.
    """

    if isinstance(axis, tuple):
        if len(axis) != 1:
            raise ValueError(
                f"Only a single axis is supported for groupby_reduction: {axis}"
            )
        axis = axis[0]

    # make sure 'by' has corresponding blocks to 'x'
    for ax in range(x.ndim):
        if ax != axis:
            by = expand_dims(by, axis=ax)
    by_chunks = tuple(
        c if i == axis else (1,) * x.numblocks[i] for i, c in enumerate(by.chunks)
    )
    by_shape = tuple(map(sum, by_chunks))
    by = broadcast_to(by, by_shape, chunks=by_chunks)

    # wrapper to squeeze 'by' to undo effect of broadcast, so it looks same
    # to user supplied func
    def _group_reduction_func_wrapper(func):
        def wrapper(a, by, **kwargs):
            return func(a, nxp.squeeze(by), **kwargs)

        return wrapper

    # initial map does group reduction on each block
    chunks = tuple(
        (num_groups,) * len(c) if i == axis else c for i, c in enumerate(x.chunks)
    )
    out = map_blocks(
        _group_reduction_func_wrapper(func),
        x,
        by,
        dtype=intermediate_dtype,
        chunks=chunks,
        axis=axis,
        intermediate_dtype=intermediate_dtype,
        num_groups=num_groups,
    )

    # add a dummy dimension to reduce over
    dummy_axis = -1
    out = expand_dims(out, axis=dummy_axis)

    # then reduce across blocks
    return reduction(
        out,
        func=None,
        combine_func=combine_func,
        aggregate_func=aggregate_func,
        axis=(dummy_axis, axis),  # dummy and group axis
        intermediate_dtype=intermediate_dtype,
        dtype=dtype,
        keepdims=keepdims,
        split_every=split_every,
        combine_sizes={axis: num_groups},  # group axis doesn't have size 1
        extra_func_kwargs=dict(dtype=intermediate_dtype, dummy_axis=dummy_axis),
    )


def groupby_blockwise(
    x: "Array",
    by,
    func,
    axis=None,
    dtype=None,
    num_groups=None,
    **kwargs,
):
    """A blockwise operation that performs groupby aggregations.

    Parameters
    ----------
    x: Array
        Array being grouped along one axis.
    by: nxp.array
        Array of non-negative integers to be used as labels with which to group
        the values in ``x`` along the reduction axis. Must be a 1D array.
    func: callable
        Function to apply to each chunk of data. The output of the
        function is a chunk with size corresponding to the number of groups in the
        input chunk along the reduction axis.
    axis: int or sequence of ints, optional
        Axis to aggregate along. Only supports a single axis.
    dtype: dtype
        Data type of output.
    num_groups: int
        The number of groups in the grouping array ``by``.
    """

    if by.ndim != 1:
        raise ValueError(f"Array `by` must be 1D, but has {by.ndim} dimensions.")

    if isinstance(axis, tuple):
        if len(axis) != 1:
            raise ValueError(
                f"Only a single axis is supported for groupby_reduction: {axis}"
            )
        axis = axis[0]

    newchunks, groups_per_chunk = _get_chunks_for_groups(
        x.numblocks[axis],
        by,
        num_groups=num_groups,
    )

    # calculate the chunking used to read the input array 'x'
    read_chunks = tuple(newchunks if i == axis else c for i, c in enumerate(x.chunks))

    # 'by' is not a cubed array, but we still read it in chunks
    by_read_chunks = (newchunks,)

    # find shape and chunks for the output
    shape = tuple(num_groups if i == axis else s for i, s in enumerate(x.shape))
    chunks = tuple(
        groups_per_chunk if i == axis else c for i, c in enumerate(x.chunksize)
    )
    target_chunks = normalize_chunks(chunks, shape, dtype=dtype)

    def selection_function(out_key):
        out_coords = out_key[1:]
        block_id = out_coords
        return get_item(read_chunks, block_id)

    # in general each selection overlaps 2 input blocks
    max_num_input_blocks = 2

    return map_selection(
        _process_blockwise_chunk,
        selection_function,
        x,
        shape=shape,
        dtype=dtype,
        chunks=target_chunks,
        max_num_input_blocks=max_num_input_blocks,
        axis=axis,
        by=by,
        blockwise_func=func,
        by_read_chunks=by_read_chunks,
        target_chunks=target_chunks,
        groups_per_chunk=groups_per_chunk,
        **kwargs,
    )


def _process_blockwise_chunk(
    a,
    axis=None,
    by=None,
    blockwise_func=None,
    by_read_chunks=None,
    target_chunks=None,
    groups_per_chunk=None,
    block_id=None,
    **kwargs,
):
    idx = block_id
    bi = idx[axis]

    by = by[get_item(by_read_chunks, (bi,))]

    start_group = bi * groups_per_chunk

    return blockwise_func(
        a,
        by,
        axis=axis,
        start_group=start_group,
        num_groups=target_chunks[axis][bi],
        **kwargs,
    )


def _get_chunks_for_groups(num_chunks, labels, num_groups):
    """Find new chunking so that there are an equal number of group labels per chunk."""

    # find the start indexes of each group
    start_indexes = nxp.searchsorted(labels, nxp.arange(num_groups))

    # find the number of groups per chunk
    groups_per_chunk = max(num_groups // num_chunks, 1)

    # each chunk has groups_per_chunk groups in it (except possibly last one)
    chunk_boundaries = start_indexes[::groups_per_chunk]

    # successive differences give the new chunk sizes (include end index for last chunk)
    newchunks = nxp.diff(chunk_boundaries, append=len(labels))

    return tuple(newchunks), groups_per_chunk
