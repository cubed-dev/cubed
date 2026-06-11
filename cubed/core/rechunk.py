"""Core rechunking algorithm based on rechunker, but adapted for Cubed to support regular Zarr chunks."""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, log, prod

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
    calculate_stage_chunks,
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


def _limit_chunks(chunks, ratio_chunks, align_to, max_blocks):
    """Reduce chunks greedily so that prod(ceil(chunks[i] / ratio_chunks[i])) <= max_blocks.

    Each reduced chunk remains a positive multiple of align_to[i].

    For fan-out limiting: ratio_chunks = align_to = target_chunks, max_blocks = max_output_blocks.
    For fan-in limiting: ratio_chunks = source_chunks, align_to = store_chunks, max_blocks = max_input_blocks.
    When ratio_chunks == align_to the ceil is always exact (chunks are guaranteed multiples).

    If the constraint cannot be met then chunks is returned unchanged.
    """
    cc = list(chunks)
    rc = list(ratio_chunks)
    at = list(align_to)
    while True:
        ratios = [ceil(c / r) for c, r in zip(cc, rc)]
        total = prod(ratios)
        if total <= max_blocks:
            break
        reducible = [
            (ratios[i], i) for i in range(len(cc)) if cc[i] > at[i] and ratios[i] > 1
        ]
        if not reducible:
            break  # nothing left to reduce
        _, best = max(reducible)
        other = total // ratios[best]
        new_ratio = max(1, max_blocks // other)
        # Largest multiple of at[best] with ceil(x / rc[best]) <= new_ratio
        new_cc = (new_ratio * rc[best] // at[best]) * at[best]
        new_cc = max(new_cc, at[best])
        if new_cc >= cc[best]:
            # defensive: unreachable when chunks are multiples of align_to
            break  # pragma: no cover
        cc[best] = new_cc
    return tuple(cc)


def _fix_copy_chunks(shape, copy_chunks, target_chunks):
    # if copy chunks are bigger than target chunks in a particular axis, then
    # round them down to the largest multiple of the target so they are aligned
    return tuple(
        cc if (cc <= tc) or (cc == n) or (cc % tc == 0) else (cc // tc) * tc
        for n, cc, tc in zip(shape, copy_chunks, target_chunks)
    )


def _compute_plan_max_fan(
    stages: _MultistagePlan,
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
) -> tuple[int, int]:
    """Compute max fan-in and fan-out across all copy ops implied by a plan.

    Mirrors the copy-op structure of _rechunk_plan in ops.py: a stage with
    read_chunks == write_chunks emits one copy op; otherwise the non-last stages
    emit only a READ op, and the last stage emits a READ op and a WRITE op.
    """
    current_source = tuple(source_chunks)
    max_fan_in = 0
    max_fan_out = 0

    for i, (read_chunks, int_chunks, write_chunks) in enumerate(stages):
        last_stage = i == len(stages) - 1
        target_chunks_ = tuple(target_chunks) if last_stage else write_chunks

        if read_chunks == write_chunks:
            fan_in = prod(ceil(r / s) for r, s in zip(read_chunks, current_source))
            fan_out = prod(ceil(r / t) for r, t in zip(read_chunks, target_chunks_))
            max_fan_in = max(max_fan_in, fan_in)
            max_fan_out = max(max_fan_out, fan_out)
            current_source = target_chunks_
        else:
            # READ op
            fan_in = prod(ceil(r / s) for r, s in zip(read_chunks, current_source))
            fan_out = prod(ceil(r / c) for r, c in zip(read_chunks, int_chunks))
            max_fan_in = max(max_fan_in, fan_in)
            max_fan_out = max(max_fan_out, fan_out)
            current_source = int_chunks

            if last_stage:
                # WRITE op
                fan_in = prod(ceil(w / s) for w, s in zip(write_chunks, current_source))
                fan_out = prod(
                    ceil(w / t) for w, t in zip(write_chunks, target_chunks_)
                )
                max_fan_in = max(max_fan_in, fan_in)
                max_fan_out = max(max_fan_out, fan_out)
                current_source = target_chunks_

    return max_fan_in, max_fan_out


def multistage_bounded_rechunking_plan(
    shape: Sequence[int],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    itemsize: int,
    max_mem: int,
    consolidate_reads: bool = True,
    consolidate_writes: bool = True,
    max_input_blocks: int | None = None,
    max_output_blocks: int | None = None,
) -> _MultistagePlan:
    """Calculate an irregular rechunking plan driven by fan-in/out budgets.

    Stages are added until every copy op's fan-in and fan-out are within
    max_input_blocks and max_output_blocks respectively, then _limit_chunks
    is applied as a post-processing step to reduce any residual fan excess.
    If IO requirements start increasing before the budget is met, the last
    plan that did not increase IO is used with a warning.
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

    if consolidate_writes:
        write_chunks = consolidate_chunks(shape, target_chunks, itemsize, max_mem)
    else:
        write_chunks = tuple(target_chunks)

    if consolidate_reads:
        read_chunk_limits = []
        for sc, wc in zip(source_chunks, write_chunks):
            read_chunk_limits.append(wc if wc > sc else None)
        read_chunks = consolidate_chunks(
            shape, source_chunks, itemsize, max_mem, read_chunk_limits
        )
    else:
        read_chunks = tuple(source_chunks)

    def _postprocess(plan: _MultistagePlan) -> _MultistagePlan:
        current_source = tuple(source_chunks)
        result = []
        for i, (rc, ic, wc) in enumerate(plan):
            last_stage = i == len(plan) - 1
            tc = tuple(target_chunks) if last_stage else wc
            if rc == wc:
                cc = rc
                if max_input_blocks is not None:
                    cc = _limit_chunks(cc, current_source, tc, max_input_blocks)
                if max_output_blocks is not None:
                    cc = _limit_chunks(cc, tc, tc, max_output_blocks)
                result.append((cc, ic, cc))
                current_source = tc
            else:
                cc_read = rc
                if max_input_blocks is not None:
                    cc_read = _limit_chunks(
                        cc_read, current_source, ic, max_input_blocks
                    )
                if max_output_blocks is not None:
                    cc_read = _limit_chunks(cc_read, ic, ic, max_output_blocks)
                current_source = ic
                if last_stage:
                    cc_write = wc
                    if max_input_blocks is not None:
                        cc_write = _limit_chunks(
                            cc_write, current_source, tc, max_input_blocks
                        )
                    if max_output_blocks is not None:
                        cc_write = _limit_chunks(cc_write, tc, tc, max_output_blocks)
                    result.append((cc_read, ic, cc_write))
                    current_source = tc
                else:
                    result.append((cc_read, ic, wc))
        return result

    prev_io_ops: float | None = None
    prev_plan: _MultistagePlan | None = None

    for stage_count in range(1, MAX_STAGES):
        stage_chunks = calculate_stage_chunks(read_chunks, write_chunks, stage_count)
        pre_chunks = [read_chunks] + stage_chunks
        post_chunks = stage_chunks + [write_chunks]

        int_chunks = [
            _calculate_shared_chunks(pre, post)
            for pre, post in zip(pre_chunks, post_chunks)
        ]
        plan = _postprocess(list(zip(pre_chunks, int_chunks, post_chunks)))

        max_fan_in, max_fan_out = _compute_plan_max_fan(
            plan, source_chunks, target_chunks
        )
        fan_in_ok = max_input_blocks is None or max_fan_in <= max_input_blocks
        fan_out_ok = max_output_blocks is None or max_fan_out <= max_output_blocks
        if fan_in_ok and fan_out_ok:
            return plan

        io_ops = sum(
            calculate_single_stage_io_ops(shape, pre, post)
            for pre, post in zip(pre_chunks, post_chunks)
        )
        if prev_io_ops is not None and io_ops > prev_io_ops:
            warnings.warn(
                "Search for multi-stage rechunking plan terminated before "
                "satisfying max_input_blocks/max_output_blocks constraints due to "
                f"increasing IO requirements. Max fan-in={max_fan_in}, "
                f"max fan-out={max_fan_out}. "
                f"Consider increasing max_mem ({max_mem}) to find a more efficient plan.",
                category=ExcessiveIOWarning,
                stacklevel=2,
            )
            assert prev_plan is not None
            return prev_plan

        prev_io_ops = io_ops
        prev_plan = plan

    raise AssertionError(
        "Failed to find a feasible multi-stage rechunking plan satisfying "
        f"max_input_blocks={max_input_blocks}, max_output_blocks={max_output_blocks} "
        f"within max_mem ({max_mem}). "
        f"shape={shape}, source_chunks={source_chunks}, target_chunks={target_chunks}"
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
        read_chunk_limits: list[int | None] = []
        for sc, wc in zip(source_chunks, write_chunks):
            limit: int | None
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

    prev_io_ops: float | None = None
    prev_plan: _MultistagePlan | None = None

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


def multistage_symmetric_rechunking_plan(
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    max_input_blocks: int | None = None,
    max_output_blocks: int | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Symmetric rechunking plan.

    Treats growing and shrinking dimensions independently:

    - Growing dims (target > source): copy granularity is set to the output
      intermediate, so fan-in per dim <= max_input_blocks and fan-out = 1.
    - Shrinking dims (target < source): copy granularity is set to the input
      intermediate, so fan-in = 1 and fan-out per dim <= max_output_blocks.

    Stage count is the minimum needed to keep the total fan within budget.
    Growing dims contribute multiplicatively to fan-in; shrinking dims
    contribute multiplicatively to fan-out. The stage count is driven by
    the product of simultaneously changing dims, not the per-dim maximum,
    so the budget is honoured on the total fan at every stage.

    Reversing the rechunk (swapping source and target) produces a plan with
    identical stage count and fan-in/fan-out exactly swapped.

    Returns a list of (copy_chunks, store_chunks) pairs.
    """
    ndim = len(source_chunks)
    # None means no constraint on that side; clamp to ≥2 to keep log(budget) > 0.
    budget_in = max(max_input_blocks, 2) if max_input_blocks is not None else None
    budget_out = max(max_output_blocks, 2) if max_output_blocks is not None else None

    # Stage count is driven by the *product* of simultaneously growing or shrinking
    # dims, not the per-dim max. Growing dims contribute multiplicatively to fan-in;
    # shrinking dims contribute multiplicatively to fan-out. Using the product ensures
    # the budget is honoured on the total fan, not just each dimension individually.
    if budget_in is not None:
        total_grow = prod(t / s for s, t in zip(source_chunks, target_chunks) if t > s)
        stages_grow = ceil(log(total_grow) / log(budget_in)) if total_grow > 1 else 0
    else:
        stages_grow = 0

    if budget_out is not None:
        total_shrink = prod(
            s / t for s, t in zip(source_chunks, target_chunks) if s > t
        )
        stages_shrink = (
            ceil(log(total_shrink) / log(budget_out)) if total_shrink > 1 else 0
        )
    else:
        stages_shrink = 0

    num_stages = max(stages_grow, stages_shrink, 1)

    # Build per-dim intermediate sequence of length num_stages + 1
    # (includes source and target as endpoints, intermediates in between).
    # Growing dims: multiples of source, so fan-in = ceil(next/prev) is exact.
    # Shrinking dims: multiples of target, so the final stage aligns to target.
    sequences = []
    for s, t in zip(source_chunks, target_chunks):
        if t > s:
            seq = [s] + multspace(s, t, num_stages - 1) + [t]
        elif t < s:
            seq = [s] + list(reversed(multspace(t, s, num_stages - 1))) + [t]
        else:
            seq = [s] * (num_stages + 1)
        sequences.append(seq)

    result = []
    for k in range(num_stages):
        prev_inter = tuple(seq[k] for seq in sequences)
        next_inter = tuple(seq[k + 1] for seq in sequences)
        # Growing dim: copy = next side (fan-in bounded, fan-out = 1).
        # Shrinking dim: copy = prev side (fan-in = 1, fan-out bounded).
        # Equal dim: copy = prev = next = source = target.
        copy_chunks = tuple(
            next_inter[i] if target_chunks[i] > source_chunks[i] else prev_inter[i]
            for i in range(ndim)
        )
        result.append((copy_chunks, next_inter))
    return result


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

    source_aligned: bool | None = None
    """Are copy chunks aligned with source chunks?"""

    target_aligned: bool | None = None
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
    max_task_input_blocks: int
    max_task_output_blocks: int

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
        num_task_input_blocks = [
            d["pipeline"].config.num_input_blocks[0] for _, d in rechunks
        ]
        num_task_output_blocks = [
            d["pipeline"].config.num_output_blocks[0] for _, d in rechunks
        ]

        return cls(
            num_copy_ops=len(rechunk_plan.copy_ops),
            num_tasks=plan.num_tasks,
            total_nbytes_written=plan.total_nbytes_written,
            max_task_iops=max(num_task_iops),
            max_task_input_blocks=max(num_task_input_blocks),
            max_task_output_blocks=max(num_task_output_blocks),
        )


def rechunk_plan(
    x,
    chunks,
    *,
    min_mem=None,
    allow_irregular=True,
    max_input_blocks=None,
    max_output_blocks=None,
):
    from cubed.core.ops import _rechunk_plan

    copy_ops = []
    source_chunks = x.chunksize
    for copy_chunks, target_chunks in _rechunk_plan(
        x,
        chunks,
        min_mem=min_mem,
        allow_irregular=allow_irregular,
        max_input_blocks=max_input_blocks,
        max_output_blocks=max_output_blocks,
    ):
        copy_ops.append(RechunkCopy(x.shape, source_chunks, copy_chunks, target_chunks))
        source_chunks = target_chunks
    return RechunkPlan(copy_ops)
