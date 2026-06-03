# Rechunk IOps Improvement

**Branch:** `rechunk-iops-experiment-optimize`
**Base commit:** `8912de03` (Rechunk plan viz)

## Background

Cubed's rechunking decomposes a large array into a sequence of *copy operations*, each reading a block
of the source array (the *copy chunk*) and writing it to an intermediate or final store in smaller
pieces (the *store chunks*).  The number of S3 (or other object-store) requests a single task must
make is:

```
task_iops = fan_in + fan_out
fan_in  = product(ceil(copy_dim / source_dim) for each axis)
fan_out = product(copy_dim // store_dim for each axis)
```

For ERA5-style rechunks (time-chunked → space-chunked), the final copy op collapses many time chunks
into a single spatial chunk, producing very high fan-out values (hundreds to thousands of IOps per
task).  This matters in practice because cloud object stores throttle per-prefix request rates, and
individual Lambda functions have limited network concurrency.

## Changes

### 1. `max_iops` — per-task IOps bound (`894a7796`)

Added `max_iops: int | None` parameter to `rechunk()`, `array.rechunk()`, `_rechunk_plan()`, and
`rechunk_plan()`.

**Mechanism** — `_limit_write_chunks_fan_out(write_chunks, target_chunks, max_iops)` in
`cubed/core/rechunk.py`:

- Computes the fan-out ratio per axis (`write_dim // target_dim`).
- Greedily reduces the dimension with the largest ratio, allocating as much of the remaining
  IOps budget as possible, until `product(ratios) ≤ max_iops`.
- Each reduced dimension is rounded down to the nearest multiple of its target chunk size,
  preserving Zarr alignment.

**Application** — `_maybe_limit(copy_chunks, store_chunks)` is called inline inside `_rechunk_plan`
for every yielded `(copy_chunks, store_chunks)` pair, applying the limit uniformly to all copy ops
(not just the last), and equally to both the regular and irregular planners.

> **Alignment caveat:** because copy chunks must be exact multiples of target chunks, the achieved
> fan-out may slightly exceed `max_iops`.  Tests allow ~20% overshoot (e.g. `max_iops=50` →
> `max_task_iops ≤ 60`).

### 2. `rechunk_plans()` — Pareto-optimal plan enumeration (`5e013481`)

```python
from cubed.core.rechunk import rechunk_plans

plan_set = rechunk_plans(x, target_chunks)          # → RechunkPlanSet
plan_set = rechunk_plans(x, target_chunks, max_iops=50)
```

**Algorithm:**

1. Compute `rechunker_max_mem` from the array's spec (memory budget per stage).
2. Sweep 21 log-spaced `min_mem` values from `0` to `rechunker_max_mem`.
3. For each `min_mem`, call `rechunk_plan()` and build a `RechunkPlanStats`.  Silently skip failures.
4. Deduplicate by `num_copy_ops` (stage count), keeping the first representative for each stage count.
5. Apply a **3-objective Pareto filter** on `(num_copy_ops, total_nbytes_written, max_task_iops)`:
   a plan is kept iff no other plan is at least as good on all three and strictly better on at least one.
6. Return a `RechunkPlanSet` sorted by stage count ascending.

**`RechunkPlanSet`** (`cubed/core/rechunk.py`):

| Method | Description |
|--------|-------------|
| `.plans` | `list[(RechunkPlan, RechunkPlanStats)]` sorted by stage count |
| `.best(max_iops=None)` | Returns `(plan, stats)` minimising bytes written subject to IOps constraint |
| `._repr_html_()` | HTML table of stages/tasks/written/max_iops |

`best()` with a `max_iops` constraint: if no plan satisfies it, emits a `UserWarning` and falls back
to the plan with the lowest achievable IOps.

### 3. `rechunk(..., optimize=True)` — constrained plan selection (`5e013481`)

```python
b = a.rechunk(target_chunks, max_iops=450, optimize=True)
```

When `optimize=True`, `rechunk()` in `cubed/core/ops.py`:

1. Calls `rechunk_plans(x, chunks, allow_irregular=allow_irregular, max_iops=max_iops)`.
2. Extracts the `min_mem` for the best plan via `plan_set._best_min_mem(max_iops=max_iops)`.
3. Runs the normal `_rechunk_plan()` path with that `min_mem`.

This means the final array is computed exactly once; the plan sweep is a planning-only overhead
(no data movement).

**Default behaviour is unchanged:** `optimize=False` uses `min_mem = rechunker_max_mem // 20`
(the previous default heuristic).

### 4. `_rechunker_max_mem(x)` helper (`cubed/core/ops.py:979`)

Centralises the maximum memory available per rechunk stage:

```python
def _rechunker_max_mem(x):
    spec = x.spec
    buffer_copies = get_buffer_copies(spec)
    total_copies = 1 + buffer_copies.read + 1 + 1 + buffer_copies.write
    return (spec.allowed_mem - spec.reserved_mem) // total_copies
```

Previously this formula was duplicated between `rechunk()` and `_rechunk_plan()`; it is now shared.

### 5. Benchmark CLI (`examples/rechunk-bench.py`)

Added `--max-iops N` flag and a `print_plan_tables()` function that prints two ASCII tables:

**Table 1 — Pareto-optimal plans** (marking the default selection with `◀`):

```
  Pareto-optimal plans:
   stages │      tasks │   written │ max IOps │
  ────────┼────────────┼───────────┼──────────┼───
        2 │         79 │   19.2 GB │      459 │
        3 │        129 │   28.8 GB │      438 │ ◀
```

**Table 2 — Per-op breakdown** for the default plan, showing copy chunks, store chunks, fan_in,
fan_out, and total IOps per copy operation.

## Key design decisions

**Post-hoc fan-out limit** — `_maybe_limit` is applied after the planner runs, as a correction to
the copy chunks.  This keeps the planners themselves simple and ensures the same logic applies to
both regular and irregular planners.

**3-objective Pareto filter** — Using only (stages, bytes) would discard plans that have higher
bytes but lower IOps, which are genuinely useful when `max_iops` is constrained.  The 3-objective
filter preserves all non-dominated plans.

**`min_mem` as the plan handle** — `rechunk_plans` discovers plans by varying `min_mem`.  Rather
than storing the full plan object, `_best_min_mem()` returns the `min_mem` value so the normal
`rechunk()` path can replay it exactly.  This avoids duplicating array graph construction.

**No sweep of `allow_irregular`** — The current implementation sweeps `min_mem` for a fixed
`allow_irregular` setting.  The two planners could be combined in a future extension.

## Files changed

| File | Change |
|------|--------|
| `cubed/core/rechunk.py` | Added `_limit_write_chunks_fan_out`, `_fmt_bytes`, `_pareto_filter`, `RechunkPlanSet`, `rechunk_plans`; updated `rechunk_plan` to accept `max_iops` |
| `cubed/core/ops.py` | Added `_rechunker_max_mem`; updated `rechunk` and `_rechunk_plan` to accept `max_iops` and `optimize`; applied `_maybe_limit` uniformly |
| `cubed/core/array.py` | Updated `CoreArray.rechunk` signature to pass through `max_iops` and `optimize` |
| `cubed/tests/test_rechunk.py` | Added `test_rechunk_max_iops_era5_tiny`, `test_rechunk_plans_era5_tiny`, `test_rechunk_optimize_era5_tiny` |
| `examples/rechunk-bench.py` | Added `--max-iops` flag, `print_plan_tables`, ASCII table helpers |
