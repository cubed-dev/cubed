"""Core rechunking algorithm stuff."""
import logging
import warnings
from math import ceil, floor, prod
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .compat import lcm

logger = logging.getLogger(__name__)


def consolidate_chunks(
    shape: Sequence[int],
    chunks: Sequence[int],
    itemsize: int,
    max_mem: int,
    chunk_limits: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[int, ...]:
    """
    Consolidate input chunks up to a certain memory limit. Consolidation starts on the
    highest axis and proceeds towards axis 0.

    Parameters
    ----------
    shape : Tuple
        Array shape
    chunks : Tuple
        Original chunk shape (must be in form (5, 10, 20), no irregular chunks)
    max_mem : Int
        Maximum permissible chunk memory size, measured in units of itemsize
    chunk_limits : Tuple, optional
        Maximum size of each chunk along each axis. If None, don't consolidate
        axis. If -1, no limit.

    Returns
    -------
    new_chunks : tuple
        The new chunks, size guaranteed to be <= mam_mem
    """

    ndim = len(shape)
    if chunk_limits is None:
        chunk_limits = shape
    assert len(chunk_limits) == ndim

    # now convert chunk_limits to a dictionary
    # key: axis, value: limit
    chunk_limit_per_axis = {}
    for n_ax, cl in enumerate(chunk_limits):
        if cl is not None:
            if cl == -1:
                chunk_limit_per_axis[n_ax] = shape[n_ax]
            elif chunks[n_ax] <= cl <= shape[n_ax]:
                chunk_limit_per_axis[n_ax] = cl
            elif cl > shape[n_ax]:
                chunk_limit_per_axis[n_ax] = shape[n_ax]
            else:
                raise ValueError(f"Invalid chunk_limits {chunk_limits}.")

    chunk_mem = itemsize * prod(chunks)
    if chunk_mem > max_mem:
        raise ValueError(f"chunk_mem {chunk_mem} > max_mem {max_mem}")
    headroom = max_mem / chunk_mem
    logger.debug(f"  initial headroom {headroom}")

    new_chunks = list(chunks)
    # only consolidate over these axes
    axes = sorted(chunk_limit_per_axis.keys())[::-1]
    for n_axis in axes:
        upper_bound = min(shape[n_axis], chunk_limit_per_axis[n_axis])
        # try to just increase the chunk to the upper bound
        new_chunks[n_axis] = upper_bound
        chunk_mem = itemsize * prod(new_chunks)
        upper_bound_headroom = max_mem / chunk_mem
        if upper_bound_headroom > 1:
            # ok it worked
            headroom = upper_bound_headroom
            logger.debug("  ! maxed out headroom")
        else:
            # nope, that was too much
            # instead increase it by an integer multiple
            larger_chunk = int(chunks[n_axis] * int(headroom))
            # not sure the min check is needed any more; it safeguards against making it too big
            new_chunks[n_axis] = min(larger_chunk, upper_bound)
            chunk_mem = itemsize * prod(new_chunks)
            headroom = max_mem / chunk_mem

        logger.debug(f"  axis {n_axis}, {chunks[n_axis]} -> {new_chunks[n_axis]}")
        logger.debug(f"  chunk_mem {chunk_mem}, headroom {headroom}")

        assert headroom >= 1

    return tuple(new_chunks)


def _calculate_shared_chunks(
    read_chunks: Sequence[int], write_chunks: Sequence[int]
) -> Tuple[int, ...]:
    # Intermediate chunks are the smallest possible chunks which fit
    # into both read_chunks and write_chunks.
    # Example:
    #   read_chunks:            (20, 5)
    #   target_chunks:          (4, 25)
    #   intermediate_chunks:    (4, 5)
    # We don't need to check their memory usage: they are guaranteed to be smaller
    # than both read and write chunks.
    return tuple(
        min(c_read, c_target) for c_read, c_target in zip(read_chunks, write_chunks)
    )


def calculate_stage_chunks(
    read_chunks: Tuple[int, ...],
    write_chunks: Tuple[int, ...],
    stage_count: int = 1,
) -> List[Tuple[int, ...]]:
    """
    Calculate chunks after each stage of a multi-stage rechunking.

    Each stage consists of "split" step followed by a "consolidate" step.

    The strategy used here is to progressively enlarge or shrink chunks along
    each dimension by the same multiple in each stage (geometric spacing). This
    should roughly minimize the total number of arrays resulting from "split"
    steps in a multi-stage pipeline. It also keeps the total number of elements
    in each chunk constant, up to rounding error, so memory usage should remain
    constant.

    Examples::

        >>> calculate_stage_chunks((1_000_000, 1), (1, 1_000_000), stage_count=2)
        [(1000, 1000)]
        >>> calculate_stage_chunks((1_000_000, 1), (1, 1_000_000), stage_count=3)
        [(10000, 100), (100, 10000)]
        >>> calculate_stage_chunks((1_000_000, 1), (1, 1_000_000), stage_count=4)
        [(31623, 32), (1000, 1000), (32, 31623)]

    TODO: consider more sophisticated algorithms. In particular, exact geometric
    spacing often requires irregular intermediate chunk sizes, which (currently)
    cannot be stored in Zarr arrays.
    """
    approx_stages = np.geomspace(read_chunks, write_chunks, num=stage_count + 1)
    return [tuple(floor(c) for c in stage) for stage in approx_stages[1:-1]]


def _count_intermediate_chunks(source_chunk: int, target_chunk: int, size: int) -> int:
    """Count intermediate chunks required for rechunking along a dimension.

    Intermediate chunks must divide both the source and target chunks, and in
    general do not need to have a regular size. The number of intermediate
    chunks is proportional to the number of required read/write operations.

    For example, suppose we want to rechunk an array of size 20 from size 5
    chunks to size 7 chunks. We can draw out how the array elements are divided:
        0 1 2 3 4|5 6 7 8 9|10 11 12 13 14|15 16 17 18 19   (4 chunks)
        0 1 2 3 4 5 6|7 8 9 10 11 12 13|14 15 16 17 18 19   (3 chunks)

    To transfer these chunks, we would need to divide the array into irregular
    intermediate chunks that fit into both the source and target:
       0 1 2 3 4|5 6|7 8 9|10 11 12 13|14|15 16 17 18 19    (6 chunks)

    This matches what ``_count_intermediate_chunks()`` calculates::

        >>> _count_intermediate_chunks(5, 7, 20)
        6
    """
    multiple = lcm(source_chunk, target_chunk)
    splits_per_lcm = multiple // source_chunk + multiple // target_chunk - 1
    lcm_count, remainder = divmod(size, multiple)
    if remainder:
        splits_in_remainder = (
            ceil(remainder / source_chunk) + ceil(remainder / target_chunk) - 1
        )
    else:
        splits_in_remainder = 0
    return lcm_count * splits_per_lcm + splits_in_remainder


def calculate_single_stage_io_ops(
    shape: Sequence[int], in_chunks: Sequence[int], out_chunks: Sequence[int]
) -> int:
    """Count the number of read/write operations required for rechunking."""
    return prod(map(_count_intermediate_chunks, in_chunks, out_chunks, shape))


# not a tight upper bound, but ensures that the loop in
# multistage_rechunking_plan always terminates.
MAX_STAGES = 100


_MultistagePlan = List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]


class ExcessiveIOWarning(Warning):
    pass


def multistage_rechunking_plan(
    shape: Sequence[int],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    itemsize: int,
    min_mem: int,
    max_mem: int,
    consolidate_reads: bool = True,
    consolidate_writes: bool = True,
) -> _MultistagePlan:
    """Caculate a rechunking plan that can use multiple split/consolidate steps.

    For best results, max_mem should be significantly larger than min_mem (e.g.,
    10x). Otherwise an excessive number of rechunking steps will be required.
    """

    ndim = len(shape)
    if len(source_chunks) != ndim:
        raise ValueError(f"source_chunks {source_chunks} must have length {ndim}")
    if len(target_chunks) != ndim:
        raise ValueError(f"target_chunks {target_chunks} must have length {ndim}")

    source_chunk_mem = itemsize * prod(source_chunks)
    target_chunk_mem = itemsize * prod(target_chunks)

    if source_chunk_mem > max_mem:
        raise ValueError(
            f"Source chunk memory ({source_chunk_mem}) exceeds max_mem ({max_mem})"
        )
    if target_chunk_mem > max_mem:
        raise ValueError(
            f"Target chunk memory ({target_chunk_mem}) exceeds max_mem ({max_mem})"
        )

    if max_mem < min_mem:  # basic sanity check
        raise ValueError(
            f"max_mem ({max_mem}) cannot be smaller than min_mem ({min_mem})"
        )

    if consolidate_writes:
        logger.debug(
            f"consolidate_write_chunks({shape}, {target_chunks}, {itemsize}, {max_mem})"
        )
        write_chunks = consolidate_chunks(shape, target_chunks, itemsize, max_mem)
    else:
        write_chunks = tuple(target_chunks)

    if consolidate_reads:
        read_chunk_limits: List[Optional[int]] = []
        for sc, wc in zip(source_chunks, write_chunks):
            limit: Optional[int]
            if wc > sc:
                # consolidate reads over this axis, up to the write chunk size
                limit = wc
            else:
                # don't consolidate reads over this axis
                limit = None
            read_chunk_limits.append(limit)

        logger.debug(
            f"consolidate_read_chunks({shape}, {source_chunks}, {itemsize}, {max_mem}, {read_chunk_limits})"
        )
        read_chunks = consolidate_chunks(
            shape, source_chunks, itemsize, max_mem, read_chunk_limits
        )
    else:
        read_chunks = tuple(source_chunks)

    prev_io_ops: Optional[float] = None
    prev_plan: Optional[_MultistagePlan] = None

    # increase the number of stages until min_mem is exceeded
    for stage_count in range(1, MAX_STAGES):

        stage_chunks = calculate_stage_chunks(read_chunks, write_chunks, stage_count)
        pre_chunks = [read_chunks] + stage_chunks
        post_chunks = stage_chunks + [write_chunks]

        int_chunks = [
            _calculate_shared_chunks(pre, post)
            for pre, post in zip(pre_chunks, post_chunks)
        ]
        plan = list(zip(pre_chunks, int_chunks, post_chunks))

        int_mem = min(itemsize * prod(chunks) for chunks in int_chunks)
        if int_mem >= min_mem:
            return plan  # success!

        io_ops = sum(
            calculate_single_stage_io_ops(shape, pre, post)
            for pre, post in zip(pre_chunks, post_chunks)
        )
        if prev_io_ops is not None and io_ops > prev_io_ops:
            warnings.warn(
                "Search for multi-stage rechunking plan terminated before "
                "achieving the minimum memory requirement due to increasing IO "
                f"requirements. Smallest intermediates have size {int_mem}. "
                f"Consider decreasing min_mem ({min_mem}) or increasing "
                f"({max_mem}) to find a more efficient plan.",
                category=ExcessiveIOWarning,
            )
            assert prev_plan is not None
            return prev_plan

        prev_io_ops = io_ops
        prev_plan = plan

    raise AssertionError(
        "Failed to find a feasible multi-staging rechunking scheme satisfying "
        f"min_mem ({min_mem}) and max_mem ({max_mem}) constraints. "
        "Please file a bug report on GitHub: "
        "https://github.com/pangeo-data/rechunker/issues\n\n"
        "Include the following debugging info:\n"
        f"shape={shape}, source_chunks={source_chunks}, "
        f"target_chunks={target_chunks}, itemsize={itemsize}, "
        f"min_mem={min_mem}, max_mem={max_mem}, "
        f"consolidate_reads={consolidate_reads}, "
        f"consolidate_writes={consolidate_writes}"
    )


def rechunking_plan(
    shape: Sequence[int],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    itemsize: int,
    max_mem: int,
    consolidate_reads: bool = True,
    consolidate_writes: bool = True,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Calculate a plan for rechunking arrays.

    Parameters
    ----------
    shape : Tuple
        Array shape
    source_chunks : Tuple
        Original chunk shape (must be in form (5, 10, 20), no irregular chunks)
    target_chunks : Tuple
        Target chunk shape (must be in form (5, 10, 20), no irregular chunks)
    itemsize: int
        Number of bytes used to represent a single array element
    max_mem : int
        Maximum permissible chunk memory size, measured in units of itemsize
    consolidate_reads: bool, optional
        Whether to apply read chunk consolidation
    consolidate_writes: bool, optional
        Whether to apply write chunk consolidation

    Returns
    -------
    new_chunks : tuple
        The new chunks, size guaranteed to be <= mam_mem
    """
    (stage,) = multistage_rechunking_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize=itemsize,
        min_mem=itemsize,
        max_mem=max_mem,
        consolidate_writes=consolidate_writes,
        consolidate_reads=consolidate_reads,
    )
    return stage
