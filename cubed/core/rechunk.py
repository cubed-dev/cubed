"""Core rechunking algorithm based on rechunker, but adapted for Cubed to support regular Zarr chunks."""

import logging
import warnings
from dataclasses import dataclass
from math import floor, prod
from typing import List, Optional, Sequence

import numpy as np

from cubed.core.plan import FinalizedPlan
from cubed.types import T_RegularChunks, T_Shape
from cubed.utils import (
    normalize_chunks,
)
from cubed.vendor.rechunker.algorithm import (
    MAX_STAGES,
    ExcessiveIOWarning,
    _calculate_shared_chunks,
    _MultistagePlan,
    calculate_single_stage_io_ops,
    consolidate_chunks,
)

logger = logging.getLogger(__name__)


def verify_chunk_compatibility(
    shape,
    write_chunks,
    target_chunks,
):
    for n, wc, tc in zip(shape, write_chunks, target_chunks):
        assert (wc == n) or (wc % tc == 0), (
            f"write chunks {write_chunks} do not evenly slice target chunks {target_chunks}, "
            f"since {wc} is not a multiple of {tc}"
        )


def multspace(start: int, stop: int, num: int, endpoints: bool = False):
    """
    Returns numbers that are roughly evenly-spaced along a log scale,
    and where each number is an exact multiple of the smallest.

    Note that start and stop endpoints are not returned.

    The returned values will always be an exact multiple of the smaller
    of start and stop. But the larger of start and stop will not necessarily
    be a multiple of any of the returned values.

    Examples::

        >>> multspace(1, 1000, 2)
        [10, 100]
        >>> multspace(1000, 1, 2)
        [100, 10]
        >>> multspace(24, 43800, 3)
        [168, 1008, 7056]
    """

    if start < 1:
        raise NotImplementedError(f"start must be 1 or more, but was {start}")

    if stop < 1:
        raise NotImplementedError(f"stop must be 1 or more, but was {stop}")

    if num < 0:
        raise NotImplementedError(f"num must be positive, but was {num}")

    if endpoints:
        raise NotImplementedError("endpoints is not supported in multspace")

    if start > stop:
        return list(reversed(multspace(stop, start, num)))

    return list(_multspace(start, stop, num))[1:-1]


def _multspace(start, stop, num):
    vals = np.geomspace(start, stop, num + 2)
    vint = 1
    for v in vals:
        # next value is int multiple closest to geomspace value (rounded down)
        vint = max(floor(v / vint) * vint, 1)
        yield vint


def calculate_regular_stage_chunks(
    read_chunks: tuple[int, ...],
    write_chunks: tuple[int, ...],
    stage_count: int = 1,
) -> list[tuple[int, ...]]:
    """
    Calculate chunks after each stage of a multi-stage rechunking.

    Unlike `calculate_stage_chunks` in rechunker, this implementation
    always returns intermediate chunks sizes that work with regularly
    chunked Zarr arrays.
    """
    stages = []
    for rc, wc in zip(read_chunks, write_chunks):
        stages.append(multspace(rc, wc, num=stage_count - 1))
    return [tuple(chunks) for chunks in np.array(stages).T.tolist()]


def _limit_write_chunks_fan_out(write_chunks, target_chunks, max_output_blocks):
    """Reduce write_chunks so that fan-out (number of target chunks per write chunk) is at most max_output_blocks.

    write_chunks must be multiples of target_chunks in each dimension (guaranteed by consolidate_chunks).
    Greedily reduces the dimension with the largest ratio until the overall fan-out is within budget.
    """
    wc = list(write_chunks)
    tc = list(target_chunks)
    while True:
        ratios = [w // t for w, t in zip(wc, tc)]
        fan_out = prod(ratios)
        if fan_out <= max_output_blocks:
            break
        best = max(range(len(wc)), key=lambda i: ratios[i])
        if ratios[best] <= 1:
            break  # already at minimum in every dimension
        # Allocate as much of the remaining budget as possible to this dimension
        other = fan_out // ratios[best]
        new_ratio = max(1, max_output_blocks // other)
        wc[best] = new_ratio * tc[best]
    return tuple(wc)


def _fix_copy_chunks(shape, copy_chunks, target_chunks):
    # if copy chunks are bigger than target chunks in a particular axis, then
    # round them down to the largest multiple of the target so they are aligned
    return tuple(
        cc if (cc <= tc) or (cc == n) or (cc % tc == 0) else (cc // tc) * tc
        for n, cc, tc in zip(shape, copy_chunks, target_chunks)
    )


def multistage_regular_rechunking_plan(
    shape: Sequence[int],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    itemsize: int,
    min_mem: int,
    max_mem: int,
    consolidate_reads: bool = True,
    consolidate_writes: bool = True,
) -> _MultistagePlan:
    """Calculate a rechunking plan that can use multiple split/consolidate steps.

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
        stage_chunks = calculate_regular_stage_chunks(
            read_chunks, write_chunks, stage_count
        )
        # adjust read_chunks to ensure they align with following stage
        read_chunks = _fix_copy_chunks(
            shape, read_chunks, (stage_chunks + [write_chunks])[0]
        )
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
                stacklevel=2,
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


@dataclass
class RechunkCopy:
    shape: T_Shape
    """The shape of the array being rechunked."""

    source_chunks: T_RegularChunks
    """The chunks of the source array for this copy operation."""

    copy_chunks: T_RegularChunks
    """The chunks used for a single task for this copy operation."""

    target_chunks: T_RegularChunks
    """The chunks of the target array for this copy operation."""

    source_aligned: Optional[bool] = None
    """Are copy chunks aligned with source chunks?"""

    target_aligned: Optional[bool] = None
    """
    Are copy chunks aligned with target chunks?
    If not then irregular chunking must be used.
    """


@dataclass
class RechunkPlan:
    copy_ops: list[RechunkCopy]

    def _repr_html_(self):
        from cubed.diagnostics.widgets import get_template
        from cubed.vendor.dask.array.svg import svg

        table = []
        for copy_op in self.copy_ops:
            row = (
                copy_op,
                svg(normalize_chunks(copy_op.source_chunks, shape=copy_op.copy_chunks)),
                svg(normalize_chunks(copy_op.target_chunks, shape=copy_op.copy_chunks)),
            )
            table.append(row)

        return get_template("rechunk_plan.j2").render(table=table)


@dataclass
class RechunkPlanStats:
    num_copy_ops: int
    num_tasks: int
    total_nbytes_written: int
    max_task_iops: int

    @classmethod
    def from_plan(cls, rechunk_plan: RechunkPlan, plan: FinalizedPlan):
        rechunks = [
            (n, d)
            for (n, d) in plan.dag.nodes(data=True)
            if d.get("op_name", None) == "rechunk"
        ]
        num_task_iops = [
            d["pipeline"].config.num_input_blocks[0]
            + d["pipeline"].config.num_output_blocks[0]
            for _, d in rechunks
        ]

        return cls(
            num_copy_ops=len(rechunk_plan.copy_ops),
            num_tasks=plan.num_tasks,
            total_nbytes_written=plan.total_nbytes_written,
            max_task_iops=max(num_task_iops),
        )


def rechunk_plan(x, chunks, *, min_mem=None, allow_irregular=False, max_iops=None):
    from cubed.core.ops import _rechunk_plan

    copy_ops = []
    source_chunks = x.chunksize
    for copy_chunks, target_chunks in _rechunk_plan(
        x, chunks, min_mem=min_mem, allow_irregular=allow_irregular, max_iops=max_iops
    ):
        copy_ops.append(RechunkCopy(x.shape, source_chunks, copy_chunks, target_chunks))
        source_chunks = target_chunks
    return RechunkPlan(copy_ops)
