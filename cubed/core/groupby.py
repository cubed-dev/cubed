from typing import TYPE_CHECKING

from cubed.array_api.manipulation_functions import broadcast_to, expand_dims
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import map_blocks, reduction_new

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
    """A reduction that performs groupby aggregations.

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
    return reduction_new(
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
