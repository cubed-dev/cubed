"""Core rechunking algorithm based on rechunker, but adapted for Cubed to support regular Zarr chunks."""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, prod

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
    allow_irregular=False,
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


def _fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"


def _pareto_filter(entries):
    """Return non-dominated entries in (num_stages, total_nbytes_written, max_task_iops) space.

    A plan is kept iff no other plan is at least as good on all three objectives
    and strictly better on at least one.
    """
    pareto = []
    for i, (_, s_i, _) in enumerate(entries):
        dominated = any(
            s_j.num_copy_ops <= s_i.num_copy_ops
            and s_j.total_nbytes_written <= s_i.total_nbytes_written
            and s_j.max_task_iops <= s_i.max_task_iops
            and (
                s_j.num_copy_ops < s_i.num_copy_ops
                or s_j.total_nbytes_written < s_i.total_nbytes_written
                or s_j.max_task_iops < s_i.max_task_iops
            )
            for j, (_, s_j, _) in enumerate(entries)
            if j != i
        )
        if not dominated:
            pareto.append(entries[i])
    return sorted(pareto, key=lambda e: e[1].num_copy_ops)


class RechunkPlanSet:
    """A set of Pareto-optimal rechunk plans trading off stages against bytes written."""

    def __init__(self, entries):
        # entries: list of (RechunkPlan, RechunkPlanStats, min_mem), sorted by num_copy_ops
        self._entries = entries

    @property
    def plans(self):
        """List of (RechunkPlan, RechunkPlanStats) tuples, sorted by num_copy_ops."""
        return [(p, s) for p, s, _ in self._entries]

    def best(self, max_input_blocks=None, max_output_blocks=None):
        """Return (plan, stats) with minimum bytes written satisfying the block constraints.

        If no plan satisfies the constraints, warns and returns the plan with
        the lowest achievable total IOps.
        """
        p, s, _ = self._best_entry(
            max_input_blocks=max_input_blocks, max_output_blocks=max_output_blocks
        )
        return p, s

    def _best_min_mem(self, max_input_blocks=None, max_output_blocks=None):
        _, _, min_mem = self._best_entry(
            max_input_blocks=max_input_blocks, max_output_blocks=max_output_blocks
        )
        return min_mem

    def _best_entry(self, max_input_blocks=None, max_output_blocks=None):
        candidates = self._entries
        if max_input_blocks is not None or max_output_blocks is not None:

            def satisfies(stats):
                if (
                    max_input_blocks is not None
                    and stats.max_task_input_blocks > max_input_blocks
                ):
                    return False
                if (
                    max_output_blocks is not None
                    and stats.max_task_output_blocks > max_output_blocks
                ):
                    return False
                return True

            satisfying = [e for e in candidates if satisfies(e[1])]
            if not satisfying:
                best_achievable = min(candidates, key=lambda e: e[1].max_task_iops)
                constraints = []
                if max_input_blocks is not None:
                    constraints.append(f"max_input_blocks={max_input_blocks}")
                if max_output_blocks is not None:
                    constraints.append(f"max_output_blocks={max_output_blocks}")
                warnings.warn(
                    f"No rechunk plan satisfies {', '.join(constraints)}. "
                    f"Falling back to plan with lowest IOps "
                    f"({best_achievable[1].max_task_iops}).",
                    UserWarning,
                    stacklevel=3,
                )
                return best_achievable
            candidates = satisfying
        return min(candidates, key=lambda e: e[1].total_nbytes_written)

    def _repr_html_(self):
        rows = []
        for _, stats, _ in self._entries:
            rows.append(
                f"<tr>"
                f"<td>{stats.num_copy_ops}</td>"
                f"<td>{stats.num_tasks:,}</td>"
                f"<td>{_fmt_bytes(stats.total_nbytes_written)}</td>"
                f"<td>{stats.max_task_input_blocks}</td>"
                f"<td>{stats.max_task_output_blocks}</td>"
                f"</tr>"
            )
        header = (
            "<thead><tr>"
            "<th>Stages</th><th>Tasks</th><th>Written</th>"
            "<th>Max In</th><th>Max Out</th>"
            "</tr></thead>"
        )
        return f"<table>{header}<tbody>{''.join(rows)}</tbody></table>"


def rechunk_plans(
    x, chunks, *, allow_irregular=False, max_input_blocks=None, max_output_blocks=None
):
    """Return a set of Pareto-optimal rechunk plans for the given array and target chunks.

    Sweeps ``min_mem`` to discover plans with different stage counts, then
    returns the non-dominated subset trading off stages against bytes written.

    Parameters
    ----------
    x : cubed.Array
        The source array.
    chunks : tuple
        The desired chunks after rechunking.
    allow_irregular : bool, optional
        If True, use the irregular rechunk planner. Default is False.
    max_input_blocks : int, optional
        If given, limit the fan-in (source blocks read per task) for all copy operations.
    max_output_blocks : int, optional
        If given, limit the fan-out (target blocks written per task) for all copy operations.

    Returns
    -------
    RechunkPlanSet
    """
    from cubed.core.ops import _rechunker_max_mem
    from cubed.core.ops import rechunk as ops_rechunk

    rechunker_max_mem = _rechunker_max_mem(x)

    # Log-spaced sweep including 0 (fewest stages) up to rechunker_max_mem (most stages)
    min_mem_values = [0] + sorted(
        set(int(v) for v in np.geomspace(1, rechunker_max_mem, 20))
    )

    seen = {}  # num_copy_ops -> (RechunkPlan, RechunkPlanStats, min_mem)
    for min_mem in min_mem_values:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rplan = rechunk_plan(
                    x,
                    chunks,
                    min_mem=min_mem,
                    allow_irregular=allow_irregular,
                    max_input_blocks=max_input_blocks,
                    max_output_blocks=max_output_blocks,
                )
                b = ops_rechunk(
                    x,
                    chunks,
                    min_mem=min_mem,
                    allow_irregular=allow_irregular,
                    max_input_blocks=max_input_blocks,
                    max_output_blocks=max_output_blocks,
                )
                stats = RechunkPlanStats.from_plan(rplan, b.plan())
        except Exception:
            continue

        key = stats.num_copy_ops
        if key not in seen:
            seen[key] = (rplan, stats, min_mem)

    entries = [seen[k] for k in sorted(seen)]
    entries = _pareto_filter(entries)
    return RechunkPlanSet(entries)
