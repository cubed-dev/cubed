# Symmetric rechunking planner

## Problem

Rechunking — changing the chunk layout of an array — is often the most
expensive step in a data pipeline. The core cost metric is the *fan*: how many
input chunks must a single task read (fan-in) or write into (fan-out). High
fan means large task memory, slow scheduling, and hard-to-parallelize I/O.

Cubed controls fan through a budget parameter `max_input_blocks` /
`max_output_blocks`. Given a budget `B`, the planner must choose a sequence of
intermediate chunk shapes such that no task ever reads more than `B` input
chunks or writes into more than `B` output chunks.

### Why a naïve bounded planner fails

The previous approach (`multistage_bounded_rechunking_plan`) built a geometric
sequence of intermediate shapes and then applied `_limit_chunks` to shrink
them post-hoc until the fan budget was met.

This fails for rechunks where a dimension has large *target* chunks relative
to the source. Consider **era5-tiny-reverse**: converting a climate dataset
from space-first storage `(2480, 10, 10)` to time-first storage `(31, 721,
1440)`. The lon dimension grows from 10 → 1440. The planner's first
intermediate uses `lon_intermediate = 1440`, so each task must read
`ceil(1440 / 10) = 144` source chunks in the lon dimension alone. Because the
intermediate is also the Zarr store chunk, `_limit_chunks` cannot reduce it
below 1440 without invalidating the write alignment. The fan-in is stuck at
**432** regardless of the budget.

The forward direction — `(31, 721, 1440)` → `(2480, 10, 10)` — works fine
because lon *shrinks* (from 1440 → 10), and the planner can freely reduce the
intermediate store size to bound fan-out. The asymmetry means an identical
dataset rechunked in opposite directions has wildly different task costs.

A similar failure occurs for **chunk-transpose** — e.g. `(1, 1440)` →
`(1440, 1)` — where both dimensions are forced to change against each other.
Neither direction has any "easy" side to exploit, so the bounded planner is
stuck at fan = 1440 in both directions.

---

## Design: treat growing and shrinking dimensions independently

The key insight is that fan-in and fan-out arise from *different* sets of
dimensions:

- A dimension that **grows** (target > source) forces many source chunks to be
  *read* for each output chunk → contributes to **fan-in**.
- A dimension that **shrinks** (target < source) forces each input chunk to be
  *written* across many output chunks → contributes to **fan-out**.

The symmetric planner exploits this by choosing the copy granularity per
dimension based on which direction it is changing:

| Dimension direction | Copy granularity | Fan-in contribution | Fan-out contribution |
|---|---|---|---|
| Growing (target > source) | next intermediate | bounded per-dim | 1 |
| Shrinking (target < source) | prev intermediate | 1 | bounded per-dim |
| Unchanged | source = target | 1 | 1 |

Because growing dims contribute only to fan-in and shrinking dims contribute
only to fan-out, the two concerns are decoupled. The budget B applies
independently to each dimension in its active direction.

### Stage count

Stage count is driven by the *product* of simultaneously growing or shrinking
dimensions, not the per-dim maximum. When multiple dims grow together, their
fan-in contributions multiply; when multiple dims shrink together, their
fan-out contributions multiply. The minimum number of stages to keep the total
fan within budget is:

```
num_stages = max(
    ceil(log(total_grow_product) / log(B)),
    ceil(log(total_shrink_product) / log(B)),
    1,
)
```

where `total_grow_product = Π (t/s for growing dims)` and
`total_shrink_product = Π (s/t for shrinking dims)`.

### Intermediate sequences

For each dimension, a log-spaced (`geomspace`) sequence of intermediates is
built between source and target. Geomspace gives uniform per-stage ratios,
minimising the maximum fan at any single stage.

### Symmetry property

Reversing a rechunk (swapping source and target) swaps which dimensions are
"growing" and which are "shrinking". Because the copy granularity rule is
symmetric — growing uses the output side, shrinking uses the input side —
the resulting plan (without memory-based consolidation) has:

- The same number of stages
- Fan-in and fan-out exactly swapped at every stage

When consolidation is applied (see below), this symmetry is only approximate
because the memory layout of the source and target arrays differs.

---

## Memory-based consolidation

For large workloads the number of stages is dominated by the product of
simultaneously shrinking dimensions. `consolidate_chunks` can reduce this
product by *freezing* dimensions that already fit in memory at their full
extent.

### Mechanism

Given `shape`, `itemsize`, and `max_mem`, `consolidate_chunks` is called on
`target_chunks`. It expands chunks from the highest axis downward, filling
available memory. A dimension that reaches its full shape size is effectively
frozen — it needs no further rechunking in the main stages.

For era5-tiny forward (`source=(31, 721, 1440)`, `target=(2480, 10, 10)`,
`max_mem ≈ 350 MB`):

```
consolidate_chunks(target=(2480, 10, 10)) → effective_target=(2480, 20, 1440)
```

Lon expands from 10 → 1440 (full shape, ≈ 143 MB). Lat expands from 10 → 20
(memory limit). Time stays at 2480 (already at shape).

This changes what the main stages see:

| | Without consolidation | With consolidation |
|---|---|---|
| Shrinking dims | lat (72×), lon (144×) | lat (36×) only — lon frozen |
| total_shrink_product | 72 × 144 = 10,382 | 36 |
| stages_shrink (B=16) | ceil(log(10382)/log(16)) = 4 | ceil(log(36)/log(16)) = 2 |

### Cleanup stage

Because the main stages now end at `effective_target` rather than
`target_chunks`, a single cleanup stage is appended to write into the actual
target chunk layout. `_limit_chunks` reduces the copy granularity to keep
fan-out within budget for this final stage.

For era5-tiny forward, the cleanup stage reads full `effective_target` chunks
`(2480, 20, 1440)` and writes `target_chunks=(2480, 10, 10)`, with copy
reduced along lon by `_limit_chunks` to satisfy the budget:

```
Cleanup: copy=(2480, 20, 50), store=(2480, 10, 10)
         fan_in  = 1 (each task reads exactly one effective_target chunk)
         fan_out = ceil(20/10) × ceil(50/10) = 2 × 5 = 10 ≤ 16
```

### When consolidation is applied

Consolidation is only used when it strictly reduces total stage count (main
stages + 1 cleanup) compared to the unconsolidated plan. For small arrays or
when `max_mem` is tight, it adds an extra cleanup stage without saving any
main stages, and is skipped. Shrinking dimensions are also capped at the
source chunk size to prevent `consolidate_chunks` from expanding them past
the source, which would invert their direction.

---

## Examples

### 1. ERA5 climate data: time-first ↔ space-first (era5-tiny)

Shape `(2480, 721, 1440)`, dtype `float32` (~10 GB).

| Direction | Source chunks | Target chunks |
|---|---|---|
| Forward | `(31, 721, 1440)` | `(2480, 10, 10)` |
| Reverse | `(2480, 10, 10)` | `(31, 721, 1440)` |

**Without a fan budget** (default planner, 3 stages each direction):

| Direction | Stages | Max fan-in | Max fan-out | Max iops/task |
|---|---|---|---|---|
| Forward | 3 | 6 | 432 | 438 |
| Reverse | 3 | 432 | 5 | 437 |

The two directions have identical costs overall, but the 432-block fan in one
direction comes from opposite sides (output for forward, input for reverse).

**With `B = 16` (symmetric planner + consolidation)**:

| Direction | Stages | Max fan-in | Max fan-out |
|---|---|---|---|
| Forward | 3 | 9 | 15 |
| Reverse | 4 | 12 | 3 |

The forward direction benefits from consolidation: lon is frozen at 1440
(fits in memory), reducing the effective shrink product from 10,382 to ~36
and cutting from 4 main stages to 2 + 1 cleanup = 3 total.

The reverse direction does not benefit — the growing lat×lon product
(10,382×) dominates regardless of consolidation, so consolidation would add
a cleanup stage without saving any main stages and is skipped.

**Forward plan (B = 16) in detail:**

```
Stage 1: copy=(277, 721, 1440), store=(277, 147, 1440)
         fan_in  = ceil(277/31) = 9   (time grows: copy = next)
         fan_out = ceil(721/147) = 5  (lat shrinks: copy = prev; lon frozen)

Stage 2: copy=(2480, 147, 1440), store=(2480, 20, 1440)
         fan_in  = ceil(2480/277) = 9
         fan_out = ceil(147/20) = 8

Cleanup: copy=(2480, 20, 50), store=(2480, 10, 10)
         fan_in  = 1  (one effective_target chunk per task)
         fan_out = ceil(20/10) × ceil(50/10) = 2 × 5 = 10
```

The previous bounded planner was **unable to produce a valid plan** for the
reverse direction with any budget — fan-in was irreducibly 432 because the
lon intermediate store chunk (1440) forced each task to read 144 source
chunks. The symmetric planner reduces it to 12 (a 36× improvement), and
crucially this is now a bounded, tunable quantity rather than a hard floor.

---

### 2. Chunk transpose

Shape `(50000, 50000)`, source `(100, 50000)`, target `(50000, 100)`.
Both dimensions change direction simultaneously.

With the bounded planner at any budget B, fan = 500 in both directions
(the full row/column length divided by the narrow chunk). "Run both directions
and pick the best" gives no improvement — both are equally bad.

**Symmetric planner with B = 16:**

```
Stage 1: copy=(700, 50000), store=(700, 5600)
         fan_in  = ceil(700/100) × 1 = 7   (dim 0 grows: copy = next)
         fan_out = 1 × ceil(50000/5600) = 9 (dim 1 shrinks: copy = prev)

Stage 2: copy=(5600, 5600), store=(5600, 700)
         fan_in  = ceil(5600/700) × 1 = 8
         fan_out = 1 × ceil(5600/700) = 8

Stage 3: copy=(50000, 700), store=(50000, 100)
         fan_in  = ceil(50000/5600) × 1 = 9
         fan_out = 1 × ceil(700/100) = 7
```

Max fan-in = 9, max fan-out = 9. Both bounded well within B = 16, down from
the 500-block fan the bounded planner produced. The per-dimension separation
is what makes this possible: dim 0 (growing) contributes only to fan-in, dim 1
(shrinking) contributes only to fan-out, so they never compound.

---

### 3. 1-D: consolidation and splitting

Single-dimension rechunks show the base case.

**Consolidate `(1,)` → `(1440,)` with B = 16 (3 stages):**

```
Stage 1: copy=(11,),   store=(11,)   fi=11, fo=1
Stage 2: copy=(121,),  store=(121,)  fi=11, fo=1
Stage 3: copy=(1440,), store=(1440,) fi=12, fo=1
```

**Split `(1440,)` → `(1,)` with B = 16 (3 stages, exact mirror):**

```
Stage 1: copy=(1440,), store=(121,)  fi=1, fo=12
Stage 2: copy=(121,),  store=(11,)   fi=1, fo=11
Stage 3: copy=(11,),   store=(1,)    fi=1, fo=11
```

Fan-in and fan-out are exactly swapped, and both stay within B = 16 per stage.

---

## Known limitations

### Total fan is the product of per-dimension fans

The budget B bounds the fan *per dimension*, not the total fan across all
dimensions. When multiple dimensions simultaneously grow or shrink, their
per-dim fans multiply:

```
total_fan_out = Π (per-dim fan-out for each shrinking dim)
```

For era5-tiny without consolidation, two simultaneously shrinking dimensions
(lat and lon) would give a maximum fan-out reaching 10,382 at 1 stage. The
stage count formula uses the product to ensure the per-stage product stays
within budget, but the total product across a single stage can still exceed B.
Consolidation mitigates this by freezing dimensions that fit in memory,
removing them from the shrink product entirely.

### Integer rounding in intermediate sequences

Intermediate chunk sizes are rounded to integers. This rounding means the
exact fan at each stage can slightly exceed the naive `B^(1/num_stages)`
expected from the stage-count formula, but the geomspace distribution keeps
per-stage ratios as uniform as possible.
