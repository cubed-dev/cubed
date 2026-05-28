"""
Rechunk benchmark on Lithops (AWS Lambda + S3).

Configure with CUBED_CONFIG pointing to a cubed.yaml (see examples/lithops/aws/).
The Lambda runtime memory should match the allowed_mem in the config.

Usage:
    export CUBED_CONFIG=examples/lithops/aws
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk --dry-run
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk --skip-source
"""

import argparse
import logging
import math
import time
import warnings

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.core.rechunk import RechunkPlanStats
from cubed.core.rechunk import rechunk_plan as make_rechunk_plan
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.rich import RichProgressBar

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

WORKLOADS = {
    "era5-tiny": dict(
        shape=(2480, 721, 1440),
        source_chunks=(31, 721, 1440),
        target_chunks=(2480, 10, 10),
        description="~10 GB, 3D time→space (ERA5 pattern, small)",
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
        description="~100 GB, 3D time→space (ERA5 pattern, medium)",
    ),
    "ocean-4d": dict(
        shape=(50, 500, 1000, 1000),
        source_chunks=(50, 1, 1000, 1000),
        target_chunks=(50, 500, 10, 10),
        description="~100 GB, 4D (depth, time, lat, lon), frozen depth axis",
    ),
    "era5-large": dict(
        shape=(258540, 721, 1440),
        source_chunks=(31, 721, 1440),
        target_chunks=(258540, 10, 10),
        description="~1 TB, 3D time→space (ERA5 pattern, full scale)",
    ),
    "sim-4d": dict(
        shape=(500, 500, 1000, 1000),
        source_chunks=(1, 100, 1000, 1000),
        target_chunks=(500, 10, 10, 1000),
        description="~930 GB, 4D (time, x, y, z), frozen z axis",
    ),
}


def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_plan_stats(wl, allow_irregular):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = xp.empty(wl["shape"], dtype=xp.float32, chunks=wl["source_chunks"])
        rplan = make_rechunk_plan(a, wl["target_chunks"], allow_irregular=allow_irregular)
        b = a.rechunk(wl["target_chunks"], allow_irregular=allow_irregular)
        stats = RechunkPlanStats.from_plan(rplan, b.plan())
    size_bytes = math.prod(wl["shape"]) * 4
    print(f"  array size:    {fmt_bytes(size_bytes)}")
    print(f"  stages:        {stats.num_copy_ops}")
    print(f"  tasks:         {stats.num_tasks:,}")
    print(f"  max task IOps: {stats.max_task_iops:,}")
    print(f"  total written: {fmt_bytes(stats.total_nbytes_written)}")


def run_to_zarr(array, store, label):
    callbacks = [RichProgressBar(), HistoryCallback()]
    t0 = time.perf_counter()
    cubed.to_zarr(array, store=store, callbacks=callbacks)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workload",
        default="era5-large",
        choices=WORKLOADS,
        help="Which workload to run (default: era5-large)",
    )
    parser.add_argument(
        "--store",
        required=True,
        metavar="S3_PREFIX",
        help="S3 path prefix for source/target zarr stores, e.g. s3://my-bucket/rechunk",
    )
    parser.add_argument(
        "--allow-irregular",
        action="store_true",
        default=False,
        help="Use irregular (rechunker) staging instead of regular staging",
    )
    parser.add_argument(
        "--skip-source",
        action="store_true",
        default=False,
        help="Skip phase 1 — source zarr already exists at STORE/WORKLOAD/source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print plan stats and exit without running",
    )
    args = parser.parse_args()

    wl = WORKLOADS[args.workload]
    source_store = f"{args.store}/{args.workload}/source"
    target_store = f"{args.store}/{args.workload}/target"

    print(f"workload:        {args.workload}")
    print(f"description:     {wl['description']}")
    print(f"shape:           {wl['shape']}")
    print(f"source_chunks:   {wl['source_chunks']}")
    print(f"target_chunks:   {wl['target_chunks']}")
    print(f"allow_irregular: {args.allow_irregular}")
    print(f"source store:    {source_store}")
    print(f"target store:    {target_store}")
    print()
    print("plan stats (using allowed_mem from CUBED_CONFIG):")
    print_plan_stats(wl, args.allow_irregular)
    print()

    if args.dry_run:
        return

    t_wall = time.perf_counter()

    if not args.skip_source:
        print("phase 1: generating and writing source data...")
        source = cubed.random.random(
            wl["shape"], dtype=xp.float32, chunks=wl["source_chunks"]
        )
        t1 = run_to_zarr(source, source_store, "elapsed")
    else:
        print("phase 1: skipped (using existing source)")
        t1 = 0.0

    print()
    print("phase 2: rechunking...")
    source_array = cubed.from_zarr(source_store)
    rechunked = source_array.rechunk(
        wl["target_chunks"], allow_irregular=args.allow_irregular
    )
    t2 = run_to_zarr(rechunked, target_store, "elapsed")

    print()
    total = time.perf_counter() - t_wall
    print(f"wall time: {total:.1f}s  (source: {t1:.1f}s, rechunk: {t2:.1f}s)")
    print("task history written to ./history/")


if __name__ == "__main__":
    main()
