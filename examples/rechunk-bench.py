"""
Rechunk benchmark on Lithops (AWS Lambda + S3).

Configure with CUBED_CONFIG pointing to a cubed.yaml (see examples/lithops/aws/).
The Lambda runtime memory should match the allowed_mem in the config.

Local usage (tiny workload, no cloud config needed):
    python examples/rechunk-bench.py --workload local-test --store /tmp/rechunk-bench --data-mode deterministic --validate

Cloud usage:
    export CUBED_CONFIG=examples/lithops/aws
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk --dry-run
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk --skip-source
    python examples/rechunk-bench.py --workload era5-large --store s3://my-bucket/rechunk --data-mode deterministic --validate
"""

import argparse
import logging
import math
import time
import warnings

import numpy as np

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.core.ops import _rechunk_plan
from cubed.core.rechunk import rechunk_plans
from cubed.utils import normalize_chunks, to_chunksize
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.rich import RichProgressBar

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

WORKLOADS = {
    "local-test": dict(
        shape=(10, 8, 6),
        source_chunks=(5, 4, 3),
        target_chunks=(10, 2, 2),
        description="tiny local test (matches unit test dimensions)",
    ),
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


def _sep(widths, vert="┼", horiz="─"):
    return "  " + (f"─{horiz*widths[0]}─" + vert) + "".join(
        f"─{horiz*w}─{vert}" for w in widths[1:-1]
    ) + f"─{horiz*widths[-1]}─"


def _row(cells, widths, align):
    parts = []
    for cell, w, a in zip(cells, widths, align):
        parts.append(f" {cell:{a}{w}} ")
    return "  " + "│".join(parts)


def print_plan_tables(wl, allow_irregular, max_iops=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = xp.empty(wl["shape"], dtype=xp.float32, chunks=wl["source_chunks"])
        plan_set = rechunk_plans(
            a, wl["target_chunks"], allow_irregular=allow_irregular, max_iops=max_iops
        )
        # Default plan: min_mem=None (same kwargs, no optimize)
        default_b = a.rechunk(
            wl["target_chunks"], allow_irregular=allow_irregular, max_iops=max_iops
        )
        default_n = sum(
            1
            for _, d in default_b.plan().dag.nodes(data=True)
            if d.get("op_name") == "rechunk"
        )
        plan_ops = list(
            _rechunk_plan(
                a, wl["target_chunks"], allow_irregular=allow_irregular, max_iops=max_iops
            )
        )

    size_bytes = math.prod(wl["shape"]) * 4
    print(f"  array size: {fmt_bytes(size_bytes)}")

    # ── Pareto-optimal plans table ───────────────────────────────────────────
    print()
    print("  Pareto-optimal plans:")
    hdr = ["stages", "tasks", "written", "max IOps", ""]
    ws = [6, 10, 9, 8, 1]
    al = [">", ">", ">", ">", "<"]
    print(_row(hdr, ws, al))
    print(_sep(ws))
    for _, stats in plan_set.plans:
        marker = "◀" if stats.num_copy_ops == default_n else ""
        row = [
            str(stats.num_copy_ops),
            f"{stats.num_tasks:,}",
            fmt_bytes(stats.total_nbytes_written),
            str(stats.max_task_iops),
            marker,
        ]
        print(_row(row, ws, al))
    print("  (◀ = default selection)")

    # ── Per-op breakdown for the default plan ────────────────────────────────
    print()
    print(f"  Per-op breakdown (default plan, {default_n} stages):")
    hdr2 = ["op", "copy chunks", "store chunks", "fan_in", "fan_out", "IOps"]
    src = wl["source_chunks"]
    rows_data = []
    for i, (copy_chunks, store_chunks) in enumerate(plan_ops):
        sc = src
        if isinstance(store_chunks[0], int):
            fan_in = math.prod(math.ceil(c / s) for c, s in zip(copy_chunks, sc))
            fan_out = math.prod(c // t for c, t in zip(copy_chunks, store_chunks))
            store_str = str(store_chunks)
        else:
            fan_in = math.prod(
                math.ceil(c / max(s)) for c, s in zip(copy_chunks, sc)
            )
            fan_out = math.prod(
                math.ceil(c / max(t)) for c, t in zip(copy_chunks, store_chunks)
            )
            store_str = str(tuple(max(t) for t in store_chunks)) + " (irr)"
        rows_data.append(
            (str(i), str(copy_chunks), store_str, str(fan_in), str(fan_out), str(fan_in + fan_out))
        )
        src = (
            store_chunks
            if isinstance(store_chunks[0], int)
            else tuple(max(t) for t in store_chunks)
        )
    ws2 = [max(len(hdr2[c]), max(len(r[c]) for r in rows_data)) for c in range(6)]
    al2 = [">", "<", "<", ">", ">", ">"]
    print(_row(hdr2, ws2, al2))
    print(_sep(ws2))
    for row in rows_data:
        print(_row(row, ws2, al2))


# ── Deterministic data generation ───────────────────────────────────────────


def _wang_hash(n):
    """Bijective 32-bit Wang hash applied elementwise to a numpy array."""
    n = n.astype(np.uint32)
    n = (n ^ np.uint32(61)) ^ (n >> np.uint32(16))
    n = n + (n << np.uint32(3))
    n = n ^ (n >> np.uint32(4))
    n = n * np.uint32(0x27D4EB2D)
    n = n ^ (n >> np.uint32(15))
    return n.view(np.int32)


def _det_block(block, block_id, _shape, _chunks):
    strides = [math.prod(_shape[i + 1:]) for i in range(len(_shape))]
    flat = np.zeros(block.shape, dtype=np.int64)
    for ax, (stride, cs) in enumerate(zip(strides, _chunks)):
        origin = block_id[ax] * cs
        idx = np.arange(origin, origin + block.shape[ax], dtype=np.int64)
        flat += idx.reshape(tuple(-1 if j == ax else 1 for j in range(len(_shape)))) * stride
    return _wang_hash(flat.astype(np.uint32))


def make_deterministic_source(shape, chunks):
    """Return a lazy int32 array where each element is wang_hash(flat_index)."""
    normalized = normalize_chunks(chunks, shape=shape, dtype=np.int32)
    return cubed.map_blocks(
        _det_block, dtype=np.int32, chunks=normalized, _shape=shape, _chunks=to_chunksize(normalized)
    )


# ── Validation ───────────────────────────────────────────────────────────────

def validate_deterministic(target_store, shape, target_chunks):
    """Check every element of the rechunked target against its expected flat index."""
    print("phase 3: validating (deterministic check)...")
    target = cubed.from_zarr(target_store)
    expected = make_deterministic_source(shape, target_chunks)
    callbacks = [RichProgressBar(), HistoryCallback()]
    t0 = time.perf_counter()
    valid = bool(xp.all(xp.equal(target, expected)).compute(callbacks=callbacks))
    elapsed = time.perf_counter() - t0
    print(f"  result: {'PASSED' if valid else 'FAILED'}  ({elapsed:.1f}s)")
    return valid, elapsed


# ── I/O helper ───────────────────────────────────────────────────────────────

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
        "--max-iops",
        type=int,
        default=None,
        metavar="N",
        help="Bound per-task S3 IOps by limiting copy chunk fan-out",
    )
    parser.add_argument(
        "--data-mode",
        choices=["random", "deterministic"],
        default="random",
        help=(
            "Source data mode (default: random). "
            "'random' uses float32 uniform random data; "
            "'deterministic' uses int32 flat-index values for exact validation."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help=(
            "Validate rechunk output after phase 2 (requires --data-mode deterministic). "
            "Checks each element against its expected Wang-hashed flat index."
        ),
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
    if args.max_iops is not None:
        print(f"max_iops:        {args.max_iops}")
    print(f"data_mode:       {args.data_mode}")
    print(f"validate:        {args.validate}")
    print(f"source store:    {source_store}")
    print(f"target store:    {target_store}")
    print()
    print("plan stats (using allowed_mem from CUBED_CONFIG):")
    print_plan_tables(wl, args.allow_irregular, max_iops=args.max_iops)
    print()

    if args.dry_run:
        return

    t_wall = time.perf_counter()

    if not args.skip_source:
        print("phase 1: generating and writing source data...")
        if args.data_mode == "deterministic":
            source = make_deterministic_source(wl["shape"], wl["source_chunks"])
        else:
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
        wl["target_chunks"], allow_irregular=args.allow_irregular, max_iops=args.max_iops
    )
    t2 = run_to_zarr(rechunked, target_store, "elapsed")

    t3 = 0.0
    if args.validate:
        if args.data_mode != "deterministic":
            print("  WARNING: --validate requires --data-mode deterministic; skipping")
        else:
            print()
            valid, t3 = validate_deterministic(
                target_store, wl["shape"], wl["target_chunks"]
            )
            if not valid:
                print("  WARNING: validation FAILED — rechunk output does not match source")

    print()
    total = time.perf_counter() - t_wall
    parts = f"source: {t1:.1f}s, rechunk: {t2:.1f}s"
    if args.validate:
        parts += f", validate: {t3:.1f}s"
    print(f"wall time: {total:.1f}s  ({parts})")
    print("task history written to ./history/")


if __name__ == "__main__":
    main()
