"""Compare the symmetric rechunking planner against the bounded planner.

For each workload and fan-in/out budget, prints side-by-side metrics so you
can check whether the new multistage_symmetric_rechunking_plan() produces plans
that are no worse than the multistage_bounded_rechunking_plan().

Usage:
    python examples/compare_symmetric_vs_bounded.py
    python examples/compare_symmetric_vs_bounded.py --budgets 16 64 256
    python examples/compare_symmetric_vs_bounded.py --workload era5-tiny
"""

import argparse
import math
import warnings
from math import ceil, prod

import cubed
import cubed.array_api as xp
from cubed.core.rechunk import (
    multistage_bounded_rechunking_plan,
    multistage_symmetric_rechunking_plan,
)
from cubed.primitive.memory import get_buffer_copies

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


def _rechunker_max_mem(shape, source_chunks, target_chunks, allowed_mem):
    """Replicate how ops.py computes the memory available to the rechunker."""
    spec = cubed.Spec(allowed_mem=allowed_mem)
    buffer_copies = get_buffer_copies(spec)
    total_copies = 1 + buffer_copies.read + 1 + 1 + buffer_copies.write
    return (spec.allowed_mem - spec.reserved_mem) // total_copies


def _bounded_to_pairs(bounded_plan, source_chunks, target_chunks):
    """Convert _MultistagePlan to (copy_chunks, store_chunks) pairs.

    Mirrors the copy-op structure in _rechunk_plan in ops.py.
    """
    pairs = []
    current_source = tuple(source_chunks)
    for i, (read_chunks, int_chunks, write_chunks) in enumerate(bounded_plan):
        last_stage = i == len(bounded_plan) - 1
        target_chunks_ = tuple(target_chunks) if last_stage else write_chunks
        if read_chunks == write_chunks:
            pairs.append((read_chunks, target_chunks_))
            current_source = target_chunks_
        else:
            pairs.append((read_chunks, int_chunks))
            current_source = int_chunks
            if last_stage:
                pairs.append((write_chunks, target_chunks_))
                current_source = target_chunks_
    return pairs


def _stats_from_pairs(pairs, source_chunks, shape):
    """Compute (num_copy_ops, num_tasks, max_fan_in, max_fan_out) from pairs."""
    fi, fo = 0, 0
    num_tasks = 0
    src = tuple(source_chunks)
    for copy_chunks, store_chunks in pairs:
        n_tasks = prod(ceil(n / c) for n, c in zip(shape, copy_chunks))
        num_tasks += n_tasks
        fi = max(fi, prod(ceil(c / s) for c, s in zip(copy_chunks, src)))
        fo = max(fo, prod(ceil(c / s) for c, s in zip(copy_chunks, store_chunks)))
        src = store_chunks
    return len(pairs), num_tasks, fi, fo


def _stats_symmetric(wl, budget, allowed_mem):
    shape = wl["shape"]
    source_chunks = wl["source_chunks"]
    target_chunks = wl["target_chunks"]
    pairs = multistage_symmetric_rechunking_plan(
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        max_input_blocks=budget,
        max_output_blocks=budget,
    )
    return _stats_from_pairs(pairs, source_chunks, shape)


def _stats_bounded(wl, budget, allowed_mem):
    shape = wl["shape"]
    source_chunks = wl["source_chunks"]
    target_chunks = wl["target_chunks"]
    itemsize = 4  # float32
    max_mem = _rechunker_max_mem(shape, source_chunks, target_chunks, allowed_mem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bounded_plan = multistage_bounded_rechunking_plan(
                shape=shape,
                source_chunks=source_chunks,
                target_chunks=target_chunks,
                itemsize=itemsize,
                max_mem=max_mem,
                max_input_blocks=budget,
                max_output_blocks=budget,
            )
        except (ValueError, AssertionError):
            return None
    pairs = _bounded_to_pairs(bounded_plan, source_chunks, target_chunks)
    return _stats_from_pairs(pairs, source_chunks, shape)


def fmt_num(n):
    return f"{n:,}"


def compare_workload(name, wl, budgets):
    chunk_bytes = math.prod(wl["source_chunks"]) * 4
    allowed_mem = max(chunk_bytes * 10, 2_500_000_000)

    print(f"\n{'='*76}")
    print(f"  {name}: {wl['description']}")
    print(f"{'='*76}")
    print(
        f"  shape={wl['shape']}  source={wl['source_chunks']}  target={wl['target_chunks']}"
    )
    print(f"  allowed_mem={allowed_mem:,}")

    sub = f"  {'':>6}  {'── symmetric ──':^38}  {'── bounded ──':^38}"
    cols = (
        f"  {'':>6}  {'ops':>3} {'tasks':>8} {'in':>5} {'out':>5} {'in ok':>5} {'out ok':>5}"
        f"    {'ops':>3} {'tasks':>8} {'in':>5} {'out':>5} {'in ok':>5} {'out ok':>5}"
    )
    print()
    print(sub)
    print(cols)
    print(f"  {'-'*6}  {'-'*38}  {'-'*38}")

    any_worse = False
    for budget in budgets:
        sym = _stats_symmetric(wl, budget, allowed_mem)
        bnd = _stats_bounded(wl, budget, allowed_mem)

        def fmt(stats, budget):
            if stats is None:
                return f"{'n/a':>3} {'n/a':>8} {'n/a':>5} {'n/a':>5} {'n/a':>5} {'n/a':>5}"
            ops, tasks, fi, fo = stats
            fi_ok = "YES" if fi <= budget else f"NO"
            fo_ok = "YES" if fo <= budget else f"NO"
            return f"{ops:>3} {fmt_num(tasks):>8} {fi:>5} {fo:>5} {fi_ok:>5} {fo_ok:>5}"

        # Flag rows where symmetric is strictly worse than bounded on any metric
        worse = ""
        if sym is not None and bnd is not None:
            ops_s, tasks_s, fi_s, fo_s = sym
            ops_b, tasks_b, fi_b, fo_b = bnd
            if ops_s > ops_b or tasks_s > tasks_b or fi_s > fi_b or fo_s > fo_b:
                worse = " !"
                any_worse = True

        print(f"  {budget:>6}  {fmt(sym, budget)}    {fmt(bnd, budget)}{worse}")

    if any_worse:
        print("  (! = symmetric worse than bounded on at least one metric)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workload", choices=list(WORKLOADS) + ["all"], default="all"
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[16, 64, 256],
        metavar="B",
        help="Fan-in/out budget values to test",
    )
    args = parser.parse_args()

    workloads = (
        WORKLOADS if args.workload == "all" else {args.workload: WORKLOADS[args.workload]}
    )

    print("Comparing symmetric planner vs bounded planner")
    print(f"Budgets: {args.budgets}  (max_input_blocks = max_output_blocks = B)")
    print()
    print("Columns: ops | tasks | max fan-in | max fan-out | fi within budget | fo within budget")
    print("'!' marks rows where symmetric is worse than bounded on any metric.")

    for name, wl in workloads.items():
        compare_workload(name, wl, args.budgets)

    print()


if __name__ == "__main__":
    main()
