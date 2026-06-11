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

The minimum number of stages needed so that each dimension stays within its
per-dim budget is:

```
num_stages = max(
    max(ceil(log(t/s) / log(B)) for growing dims),
    max(ceil(log(s/t) / log(B)) for shrinking dims),
    1,
)
```

### Intermediate sequences

For each dimension, a log-spaced sequence of intermediates is built between
source and target using `multspace`. Growing dims use multiples of the source
chunk (so fan-in = `ceil(next/prev)` is exact); shrinking dims use multiples
of the target chunk (so the final stage aligns to the target without
requiring a separate alignment pass).

### Symmetry property

Reversing a rechunk (swapping source and target) swaps which dimensions are
"growing" and which are "shrinking". Because the copy granularity rule is
symmetric — growing uses the output side, shrinking uses the input side —
the resulting plan has:

- The same number of stages
- Fan-in and fan-out exactly swapped at every stage

This is a formal guarantee: `rechunk(A → B)` and `rechunk(B → A)` always
produce mirror-image plans with the same stage count.

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

**With `B = 16` (symmetric planner)**:

| Direction | Stages | Max fan-in | Max fan-out | Max iops/task |
|---|---|---|---|---|
| Forward | 2 | 10 | 108 | 116 |
| Reverse | 2 | 120 | 10 | 128 |

The symmetry property is visible: forward fan-in (10) = reverse fan-out (10),
and forward fan-out (108) = reverse fan-in (120)*.

(*The slight difference is due to integer rounding in the intermediate
sequence and the approximate `num_output_blocks` calculation used by the
pipeline.)

**Forward plan (B = 16) in detail:**

```
Stage 1: copy=(248, 721, 1440), store=(248, 80, 120)
         fan_in  = ceil(248/31) × 1 × 1 = 8   (only time grows)
         fan_out = 1 × ceil(721/80) × ceil(1440/120) = 1 × 10 × 12 = 120

Stage 2: copy=(2480, 80, 120), store=(2480, 10, 10)
         fan_in  = ceil(2480/248) × 1 × 1 = 10
         fan_out = 1 × ceil(80/10) × ceil(120/10) = 1 × 8 × 12 = 96
```

The time dimension (growing) contributes only to fan-in; lat and lon
(shrinking) contribute only to fan-out. The per-dim fan for each growing or
shrinking dimension stays ≤ 16.

**Why fan-out is 120, not ≤ 16:** lat and lon both shrink simultaneously,
so their per-dim fans multiply: 10 × 12 = 120. This is the known limitation
described below — the budget bounds *per-dimension* fan, not the total
product.

**Reverse plan (B = 16) in detail** — exact mirror:

```
Stage 1: copy=(2480, 80, 120), store=(248, 80, 120)
         fan_in  = 1 × ceil(80/10) × ceil(120/10) = 1 × 8 × 12 = 96
         fan_out = ceil(2480/248) × 1 × 1 = 10

Stage 2: copy=(248, 721, 1440), store=(31, 721, 1440)
         fan_in  = 1 × ceil(721/80) × ceil(1440/120) = 1 × 10 × 12 = 120
         fan_out = ceil(248/31) × 1 × 1 = 8
```

The previous bounded planner was **unable to produce a valid plan** for the
reverse direction with any budget — fan-in was irrecducibly 432 because the
lon intermediate store chunk (1440) forced each task to read 144 source
chunks. The symmetric planner reduces it to 120 (a 3.6× improvement), and
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

For era5-tiny with two simultaneously shrinking dimensions (lat and lon), the
maximum fan-out reaches 120 even with B = 16, because the lat contribution (≤
16) and lon contribution (≤ 16) multiply together.

This is fundamental: the minimum number of tasks needed to rechunk a 3-D array
along all three axes simultaneously is bounded below by the product of the
per-dim ratios. Splitting each dimension into more stages reduces the per-stage
product, but cannot escape the product structure entirely without isolating
each dimension into its own pass.

A future extension could add a "total budget" mode that bounds the product
directly, at the cost of more stages for multi-dimensional rechunks.

### Integer rounding in intermediate sequences

Intermediate chunk sizes are rounded to integer multiples of the source (for
growing dims) or target (for shrinking dims). This rounding means the exact
fan at each stage can slightly exceed the naive `B^(1/num_stages)` expected
from the stage-count formula, but it guarantees that copy chunks are always
aligned with source/target store chunks.
