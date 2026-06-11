# Rechunking algorithms: existing vs symmetric

This document explains how the two rechunking algorithms in Cubed work and
why they produce different plans. It uses a concrete 2-D example small enough
to draw by hand.

---

## The problem

**Rechunking** changes the chunk layout of an array without moving any data
values. Two key cost metrics are:

- **Fan-in**: how many source chunks a single task must *read*.
- **Fan-out**: how many target chunks a single task must *write*.

High fan means large memory per task, slow scheduling, and hard-to-parallelise
I/O. A budget `B` caps both.

---

## Example setup

Shape `(8, 8)`, `itemsize = 1` byte, `max_mem = 16` bytes per task, budget `B = 2`.

| | dim 0 (rows) | dim 1 (cols) |
|---|---|---|
| Source chunks | 2 | 8 |
| Target chunks | 8 | 2 |

The source layout is wide row slabs; the target is tall column slabs. Every
element must cross from one orientation to the other.

**Source** — 4 chunks of shape `(2, 8)`:

```
     col 0─────────────────7
     ┌──────────────────────┐
row 0│                      │
     │           A          │ chunk A (rows 0–1)
row 1│                      │
     ├──────────────────────┤
row 2│                      │
     │           B          │ chunk B (rows 2–3)
row 3│                      │
     ├──────────────────────┤
row 4│                      │
     │           C          │ chunk C (rows 4–5)
row 5│                      │
     ├──────────────────────┤
row 6│                      │
     │           D          │ chunk D (rows 6–7)
row 7│                      │
     └──────────────────────┘
```

**Target** — 4 chunks of shape `(8, 2)`:

```
     col 0──1  2──3  4──5  6──7
     ┌─────┬─────┬─────┬─────┐
row 0│     │     │     │     │
row 1│     │     │     │     │
row 2│  W  │  X  │  Y  │  Z  │
row 3│     │     │     │     │
row 4│     │     │     │     │
row 5│     │     │     │     │
row 6│     │     │     │     │
row 7│     │     │     │     │
     └─────┴─────┴─────┴─────┘
```

---

## The existing algorithm

The existing algorithm (`multistage_rechunking_plan`) works by finding an
*intermediate* chunk shape that makes each single-hop rechunk fit within
`max_mem`. It treats the array as a single homogeneous object and picks the
smallest intermediate that works in both directions.

For this example the intermediate is `(2, 2)` — a grid of 16 small squares:

```
     col 0─1 2─3 4─5 6─7
     ┌───┬───┬───┬───┐
row 0│ 0 │ 1 │ 2 │ 3 │
row 1│   │   │   │   │
     ├───┼───┼───┼───┤
row 2│ 4 │ 5 │ 6 │ 7 │
row 3│   │   │   │   │
     ├───┼───┼───┼───┤
row 4│ 8 │ 9 │10 │11 │
row 5│   │   │   │   │
     ├───┼───┼───┼───┤
row 6│12 │13 │14 │15 │
row 7│   │   │   │   │
     └───┴───┴───┴───┘
```

**Stage 1: source `(2, 8)` → intermediate `(2, 2)`.** Chunk shape unchanged in
rows, split by 4 in cols. Each task reads one source chunk and *scatters* it
across 4 intermediate chunks along the column axis.

```
  Source chunk A (rows 0–1, all 8 cols)
  ┌──────────────────────┐
  │          A           │  reads 1 source chunk
  └──────────────────────┘
          │
     splits into 4 intermediate chunks:
          │
  ┌───┐ ┌───┐ ┌───┐ ┌───┐
  │ 0 │ │ 1 │ │ 2 │ │ 3 │  writes 4 chunks  (fan-out = 4)
  └───┘ └───┘ └───┘ └───┘
```

Fan-in = 1, fan-out = 4 (exceeds B = 2).

**Stage 2: intermediate `(2, 2)` → target `(8, 2)`.** Column extent unchanged,
merged by 4 in rows. Each task *gathers* 4 intermediate chunks to produce one
tall column slab.

```
  ┌───┐ ┌───┐ ┌───┐ ┌───┐
  │ 0 │ │ 4 │ │ 8 │ │12 │  reads 4 chunks (fan-in = 4)
  └───┘ └───┘ └───┘ └───┘
          │
     merges into one target chunk:
          │
  ┌─────┐
  │  W  │  writes 1 chunk  (fan-out = 1)
  └─────┘
```

Fan-in = 4, fan-out = 1 (fan-in exceeds B = 2).

**Summary for existing algorithm (B = 2):**

| Stage | copy | store | tasks | fan-in | fan-out |
|---|---|---|---|---|---|
| 1 | `(2, 8)` | `(2, 2)` | 4 | 1 | **4** |
| 2 | `(8, 2)` | `(8, 2)` | 4 | **4** | 1 |

The two stages have complementary, high fan: one stage scatters (high fan-out)
and the other gathers (high fan-in). Each stage violates the budget of 2.

---

## The symmetric algorithm

The symmetric algorithm (`multistage_symmetric_rechunking_plan`) treats the two
dimensions *independently* based on the direction each is changing:

- **Growing** dimension (rows: 2 → 8): each task must read multiple source
  chunks → contributes to **fan-in**.
- **Shrinking** dimension (cols: 8 → 2): each task must write to multiple target
  chunks → contributes to **fan-out**.

Because growing dims contribute only to fan-in and shrinking dims only to
fan-out, the two fans are decoupled. Each can be controlled to within budget B
without one direction pushing the other over the limit.

**Stage count.** The grow ratio is 8 ÷ 2 = 4 and the shrink ratio is 8 ÷ 2 = 4.
With B = 2, each ratio needs ceil(log 4 / log 2) = 2 stages. So `num_stages = 2`.

**Intermediates** are chosen on a geometric (log-spaced) sequence between
source and target for each dimension:

- rows: `[2, 4, 8]` — stages advance: 2 → 4 → 8
- cols: `[8, 4, 2]` — stages advance: 8 → 4 → 2

After stage 1 the intermediate chunks are `(4, 4)`:

```
     col 0───────3  4───────7
     ┌───────────┬───────────┐
row 0│           │           │
row 1│     P     │     Q     │
row 2│           │           │
row 3│           │           │
     ├───────────┼───────────┤
row 4│           │           │
row 5│     R     │     S     │
row 6│           │           │
row 7│           │           │
     └───────────┴───────────┘
```

**Stage 1: source `(2, 8)` → intermediate `(4, 4)`.** Each task produces one
`(4, 4)` intermediate chunk. The copy granularity is also `(4, 4)` — rows use
the *next* intermediate (4, the growing side) and cols use the *previous*
intermediate capped to fit in memory (4, limited from 8 by `max_mem = 16`).

```
  Source chunks A and B (rows 0–3, all 8 cols → but limited to 4 cols per task)
  ┌──────────────────────┐
  │          A           │  reads 2 source chunks (fi = ceil(4/2)×ceil(4/8) = 2)
  │          B           │         (rows grow: read next; cols shrink: read prev)
  └──────────────────────┘
          │
  ┌───────────┐
  │     P     │  writes 1 intermediate chunk  (fan-out = 1)
  └───────────┘
```

Fan-in = 2, fan-out = 1. Both within budget B = 2.

**Stage 2: intermediate `(4, 4)` → target `(8, 2)`.** The copy granularity is
`(8, 2)` — rows use the next intermediate (8 = target) and cols use the
previous (4, limited to 2 by `max_mem = 16`).

```
  Intermediate chunks P and R (rows 0–7, cols 0–1)
  ┌───────────┐
  │     P     │  reads 2 intermediate chunks (fi = ceil(8/4)×ceil(2/4) = 2)
  │     R     │
  └───────────┘
          │
  ┌─────┐
  │  W  │  writes 1 target chunk  (fan-out = 1)
  └─────┘
```

Fan-in = 2, fan-out = 1. Both within budget B = 2.

**Summary for symmetric algorithm (B = 2):**

| Stage | copy | store | tasks | fan-in | fan-out |
|---|---|---|---|---|---|
| 1 | `(4, 4)` | `(4, 4)` | 4 | **2** | **1** |
| 2 | `(8, 2)` | `(8, 2)` | 4 | **2** | **1** |

Every stage meets the budget of 2.

---

## Side-by-side comparison

```
Existing algorithm                     Symmetric algorithm
──────────────────────────────────     ──────────────────────────────────
Source (2,8)                           Source (2,8)
  A │ B │ C │ D  (4 wide-row chunks)     A │ B │ C │ D  (4 wide-row chunks)

  Stage 1: scatter cols                  Stage 1: grow rows + shrink cols
  (2,8)→(2,2)  fo=4 ← HIGH              (4,4)→(4,4)  fi=2, fo=1

Intermediate (2,2): 16 tiny chunks     Intermediate (4,4): 4 square chunks
 0│ 1│ 2│ 3                              P │ Q
 4│ 5│ 6│ 7                              R │ S
 8│ 9│10│11
12│13│14│15

  Stage 2: gather rows                   Stage 2: grow rows + shrink cols
  (8,2)→(8,2)  fi=4 ← HIGH              (8,2)→(8,2)  fi=2, fo=1

Target (8,2)                           Target (8,2)
  W │ X │ Y │ Z  (4 tall-col chunks)     W │ X │ Y │ Z  (4 tall-col chunks)
```

| Metric | Existing | Symmetric |
|---|---|---|
| Stages | 2 | 2 |
| Tasks total | 8 | 8 |
| Intermediate chunks | 16 | 4 |
| Max fan-in | **4** | **2** |
| Max fan-out | **4** | **1** |
| Budget B = 2 met? | No | Yes |

---

## Why the difference?

The existing algorithm produces a single intermediate that is the *smallest*
shape satisfying the memory limit for each single-hop copy. This intermediate
is usually fine-grained along all dimensions, which is efficient for total
bytes written but forces alternating high-fan stages: one stage scatters and
the other gathers. Each stage has a high fan in one direction.

The symmetric algorithm keeps fan-in and fan-out *separate*. Growing
dimensions advance only during the fan-in calculation; shrinking dimensions
advance only during the fan-out calculation. The fan from each dimension is
bounded independently, so both stay within the budget at every stage. The
intermediate uses larger, squarer chunks rather than tiny sub-blocks, which
keeps the intermediate chunk count low.

The trade-off: the symmetric algorithm may use a larger intermediate (4 chunks
of `(4, 4)` vs 16 chunks of `(2, 2)` here), writing more total bytes to the
intermediate store. But this is usually acceptable because the fan savings
reduce peak memory per task and improve scheduling.

---

## Larger workloads

For high-dimensional arrays the fan separation becomes critical. Consider the
ERA5 climate dataset in its time-first storage format:

- Shape: `(2480, 721, 1440)`, float32
- Source: `(31, 721, 1440)` — time slabs covering the full lat/lon grid
- Target: `(2480, 10, 10)` — spatial tiles covering the full time series

The lat and lon dimensions both shrink (721 → 10 and 1440 → 10). Their shrink
ratios *multiply* to `72 × 144 = 10,382`. The existing algorithm produces a
fan-out up to 432 on one stage; the bounded planner cannot reduce this below
432 regardless of budget because the intermediate store chunk is locked to the
target shape. The symmetric algorithm, with memory-based consolidation to
freeze lon at full width, achieves fan-out ≤ 15 at B = 16 — a 29× improvement.

See [rechunk-symmetric-planner.md](rechunk-symmetric-planner.md) for the full
ERA5 analysis and the consolidation mechanism.
