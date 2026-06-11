"""Compare the new bounded rechunk algorithm against the Pareto optimizer.

For each workload and fan-in/out budget, prints side-by-side metrics so you
can check whether the new multistage_bounded_rechunking_plan() produces plans
that are no worse than what the Pareto optimizer selects.

Usage:
    python examples/compare_rechunk_plans.py
    python examples/compare_rechunk_plans.py --budgets 16 64 256
    python examples/compare_rechunk_plans.py --workload era5-tiny
"""

import argparse
import warnings
from math import ceil, prod

import numpy as np

import cubed
import cubed.array_api as xp
from cubed.core.ops import rechunk as ops_rechunk
from cubed.core.rechunk import RechunkPlanStats, rechunk_plan
from cubed.primitive.memory import get_buffer_copies


def _rechunker_max_mem(x):
    spec = x.spec
    buffer_copies = get_buffer_copies(spec)
    total_copies = 1 + buffer_copies.read + 1 + 1 + buffer_copies.write
    return (spec.allowed_mem - spec.reserved_mem) // total_copies

# ---------------------------------------------------------------------------
# Pareto planner (inlined from rechunk-optimize branch)
# ---------------------------------------------------------------------------


def _pareto_filter(entries):
    """Return non-dominated entries in (num_stages, total_nbytes_written, max_task_iops)."""
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
    def __init__(self, entries):
        self._entries = entries

    @property
    def plans(self):
        return [(p, s) for p, s, _ in self._entries]

    def best(self, max_input_blocks=None, max_output_blocks=None):
        candidates = self._entries
        if max_input_blocks is not None or max_output_blocks is not None:
            satisfying = [
                e for e in candidates
                if (max_input_blocks is None or e[1].max_task_input_blocks <= max_input_blocks)
                and (max_output_blocks is None or e[1].max_task_output_blocks <= max_output_blocks)
            ]
            if not satisfying:
                return None, min(candidates, key=lambda e: e[1].max_task_iops)[1]
            candidates = satisfying
        _, stats, _ = min(candidates, key=lambda e: e[1].total_nbytes_written)
        return None, stats


def _rechunk_plans_pareto(x, chunks, max_input_blocks=None, max_output_blocks=None):
    """Sweep min_mem, collect unique plans, return Pareto-filtered RechunkPlanSet."""
    rechunker_max_mem = _rechunker_max_mem(x)
    min_mem_values = [0] + sorted(
        set(int(v) for v in np.geomspace(1, rechunker_max_mem, 20))
    )

    seen = {}
    for min_mem in min_mem_values:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rplan = rechunk_plan(
                    x,
                    chunks,
                    min_mem=min_mem,
                    allow_irregular=True,
                    max_input_blocks=max_input_blocks,
                    max_output_blocks=max_output_blocks,
                )
                b = ops_rechunk(
                    x,
                    chunks,
                    min_mem=min_mem,
                    allow_irregular=True,
                    max_input_blocks=max_input_blocks,
                    max_output_blocks=max_output_blocks,
                )
                stats = RechunkPlanStats.from_plan(rplan, b.plan())
        except Exception:
            continue
        if stats.num_copy_ops not in seen:
            seen[stats.num_copy_ops] = (rplan, stats, min_mem)

    entries = [seen[k] for k in sorted(seen)]
    entries = _pareto_filter(entries)
    return RechunkPlanSet(entries)


# ---------------------------------------------------------------------------
# Workloads (from rechunk-bench branch: examples/rechunk-bench.py)
# ---------------------------------------------------------------------------

WORKLOADS = {
    "local-test": dict(
        shape=(10, 8, 6),
        source_chunks=(5, 4, 3),
        target_chunks=(10, 2, 2),
        description="tiny local test",
    ),
    "era5-tiny": dict(
        shape=(2480, 721, 1440),
        source_chunks=(31, 721, 1440),
        target_chunks=(2480, 10, 10),
        description="~10 GB, 3D time→space (ERA5, small)",
    ),
    "era5-tiny-reverse": dict(
        shape=(2480, 721, 1440),
        source_chunks=(2480, 10, 10),
        target_chunks=(31, 721, 1440),
        description="~10 GB, 3D space→time (ERA5 reverse)",
    ),
    "2d-flip": dict(
        shape=(50000, 50000),
        source_chunks=(100, 50000),
        target_chunks=(50000, 100),
        description="~10 GB, 2D row→column flip",
    ),
    "era5-medium": dict(
        shape=(25854, 721, 1440),
        source_chunks=(31, 721, 1440),
        target_chunks=(25854, 10, 10),
        description="~100 GB, 3D time→space (ERA5, medium)",
    ),
    "ocean-4d": dict(
        shape=(50, 500, 1000, 1000),
        source_chunks=(50, 1, 1000, 1000),
        target_chunks=(50, 500, 10, 10),
        description="~100 GB, 4D frozen depth axis",
    ),
    "era5-large": dict(
        shape=(258540, 721, 1440),
        source_chunks=(31, 721, 1440),
        target_chunks=(258540, 10, 10),
        description="~1 TB, 3D time→space (ERA5, full scale)",
    ),
    "sim-4d": dict(
        shape=(500, 500, 1000, 1000),
        source_chunks=(1, 100, 1000, 1000),
        target_chunks=(500, 10, 10, 1000),
        description="~930 GB, 4D frozen z axis",
    ),
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _stats_new(x, wl, budget):
    """Plan metrics from the new bounded algorithm."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rplan = rechunk_plan(
            x,
            wl["target_chunks"],
            allow_irregular=True,
            max_input_blocks=budget,
            max_output_blocks=budget,
        )
        b = ops_rechunk(
            x,
            wl["target_chunks"],
            allow_irregular=True,
            max_input_blocks=budget,
            max_output_blocks=budget,
        )
    return RechunkPlanStats.from_plan(rplan, b.plan())


def _stats_pareto(x, wl, budget):
    """Plan metrics from the Pareto optimizer, best plan for given budget."""
    plan_set = _rechunk_plans_pareto(
        x,
        wl["target_chunks"],
        max_input_blocks=budget,
        max_output_blocks=budget,
    )
    _, stats = plan_set.best(max_input_blocks=budget, max_output_blocks=budget)
    return stats


def compare_workload(name, wl, budgets):
    print(f"\n{'='*72}")
    print(f"  {name}: {wl['description']}")
    print(f"{'='*72}")
    print(f"  shape={wl['shape']}  source={wl['source_chunks']}  target={wl['target_chunks']}")

    # Size spec to at least 10x the largest chunk so rechunking plans are feasible
    import math
    chunk_bytes = math.prod(wl["source_chunks"]) * 4
    allowed_mem = max(chunk_bytes * 10, 2_500_000_000)
    print(f"  allowed_mem={allowed_mem:,}")
    spec = cubed.Spec(allowed_mem=allowed_mem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = xp.empty(wl["shape"], dtype=xp.float32, chunks=wl["source_chunks"], spec=spec)

    header = f"  {'budget':>6}  {'':^32}  {'':^32}"
    sub    = f"  {'':>6}  {'── new algorithm ──':^32}  {'── pareto best ──':^32}"
    cols   = f"  {'':>6}  {'stages':>6} {'tasks':>8} {'written':>9} {'in':>5} {'out':>5}    {'stages':>6} {'tasks':>8} {'written':>9} {'in':>5} {'out':>5}"
    print()
    print(sub)
    print(cols)
    print(f"  {'-'*6}  {'-'*32}  {'-'*32}")

    for budget in budgets:
        new_s = _stats_new(x, wl, budget)
        par_s = _stats_pareto(x, wl, budget)

        def fmt(s):
            if s is None:
                return f"{'n/a':>6} {'n/a':>8} {'n/a':>9} {'n/a':>5} {'n/a':>5}"
            return (
                f"{s.num_copy_ops:>6} {s.num_tasks:>8,} {fmt_bytes(s.total_nbytes_written):>9}"
                f" {s.max_task_input_blocks:>5} {s.max_task_output_blocks:>5}"
            )

        # flag if new is worse than pareto on any metric
        worse = par_s is not None and (
            new_s.num_copy_ops > par_s.num_copy_ops
            or new_s.total_nbytes_written > par_s.total_nbytes_written
            or new_s.max_task_input_blocks > par_s.max_task_input_blocks
            or new_s.max_task_output_blocks > par_s.max_task_output_blocks
        )
        flag = " !" if worse else ""

        print(f"  {budget:>6}  {fmt(new_s)}    {fmt(par_s)}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", choices=list(WORKLOADS) + ["all"], default="all")
    parser.add_argument("--budgets", nargs="+", type=int, default=[16, 64, 256],
                        metavar="B", help="Fan-in/out budget values to test")
    args = parser.parse_args()

    workloads = WORKLOADS if args.workload == "all" else {args.workload: WORKLOADS[args.workload]}

    print("Comparing new bounded algorithm vs Pareto optimizer")
    print(f"Budgets: {args.budgets}  (max_input_blocks = max_output_blocks = B)")
    print()
    print("Columns: stages | tasks | bytes written | max fan-in | max fan-out")
    print("'!' marks rows where new algorithm is worse than Pareto on any metric.")

    for name, wl in workloads.items():
        compare_workload(name, wl, args.budgets)

    print()


if __name__ == "__main__":
    main()
