"""
Visualise the Pareto optimisation in rechunk_plans() for the era5-tiny workload.

Produces four subplots:
  1. Sweep  — how min_mem maps to num_copy_ops (step function)
  2. Bytes vs IOps — all unique plans; Pareto frontier highlighted
  3. Stages vs Bytes — same plans from a different angle
  4. Stages vs IOps — the third 2-D projection of the 3-objective space

Run:
    python examples/rechunk-pareto-viz.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import cubed
import cubed.array_api as xp
from cubed.core.ops import _rechunker_max_mem
from cubed.core.ops import rechunk as ops_rechunk
from cubed.core.rechunk import _pareto_filter, rechunk_plan, RechunkPlanStats

# ── Workload ─────────────────────────────────────────────────────────────────

shape = (2480, 721, 1440)
source_chunks = (31, 721, 1440)
target_chunks = (2480, 10, 10)
spec = cubed.Spec(allowed_mem="2.5GB")

a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)
rechunker_max_mem = _rechunker_max_mem(a)

# ── Sweep ─────────────────────────────────────────────────────────────────────

min_mem_values = [0] + sorted(set(int(v) for v in np.geomspace(1, rechunker_max_mem, 20)))

sweep = []  # (min_mem, num_copy_ops, total_nbytes_written, max_task_input_blocks, max_task_output_blocks)
for min_mem in min_mem_values:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rplan = rechunk_plan(a, target_chunks, min_mem=min_mem)
            b = ops_rechunk(a, target_chunks, min_mem=min_mem)
            stats = RechunkPlanStats.from_plan(rplan, b.plan())
        sweep.append(
            (
                min_mem,
                stats.num_copy_ops,
                stats.total_nbytes_written,
                stats.max_task_input_blocks,
                stats.max_task_output_blocks,
            )
        )
    except Exception:
        continue

sweep_min_mem = [r[0] for r in sweep]
sweep_stages = [r[1] for r in sweep]
sweep_bytes = [r[2] for r in sweep]
sweep_in = [r[3] for r in sweep]
sweep_out = [r[4] for r in sweep]
sweep_iops = [i + o for i, o in zip(sweep_in, sweep_out)]

# ── Deduplicate by num_copy_ops ───────────────────────────────────────────────

seen = {}
for row in sweep:
    key = row[1]  # num_copy_ops
    if key not in seen:
        seen[key] = row
unique = sorted(seen.values(), key=lambda r: r[1])

u_stages = [r[1] for r in unique]
u_bytes = [r[2] for r in unique]
u_in = [r[3] for r in unique]
u_out = [r[4] for r in unique]
u_iops = [i + o for i, o in zip(u_in, u_out)]

# ── Pareto filter ─────────────────────────────────────────────────────────────
# Reconstruct lightweight stand-ins that _pareto_filter can use via .num_copy_ops,
# .total_nbytes_written, .max_task_iops.

class _S:
    def __init__(self, stages, nbytes, iops):
        self.num_copy_ops = stages
        self.total_nbytes_written = nbytes
        self.max_task_iops = iops

fake_entries = [(None, _S(r[1], r[2], r[3] + r[4]), None) for r in unique]
pareto_entries = _pareto_filter(fake_entries)
pareto_stages = {e[1].num_copy_ops for e in pareto_entries}

p_stages = [r[1] for r in unique if r[1] in pareto_stages]
p_bytes = [r[2] for r in unique if r[1] in pareto_stages]
p_in = [r[3] for r in unique if r[1] in pareto_stages]
p_out = [r[4] for r in unique if r[1] in pareto_stages]
p_iops = [i + o for i, o in zip(p_in, p_out)]

# ── Plot ──────────────────────────────────────────────────────────────────────

GB = 1e9
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    f"Pareto optimisation — era5-tiny\n"
    f"shape={shape}  source={source_chunks}  target={target_chunks}",
    fontsize=11,
)

cmap = plt.cm.tab10
stage_set = sorted(set(sweep_stages))
colors = {s: cmap(i / max(len(stage_set) - 1, 1)) for i, s in enumerate(stage_set)}

# ── 1. Sweep: min_mem → num_copy_ops ─────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(
    [m / GB for m in sweep_min_mem],
    sweep_stages,
    c=[colors[s] for s in sweep_stages],
    zorder=3,
    s=60,
)
ax.set_xscale("log")
ax.set_xlabel("min_mem (GB, log scale)")
ax.set_ylabel("num_copy_ops (stages)")
ax.set_title("1. Sweep: min_mem → stage count")
ax.set_yticks(stage_set)
ax.grid(True, alpha=0.3)

# ── 2. Bytes vs IOps ─────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.scatter(
    [b / GB for b in u_bytes],
    u_iops,
    c=[colors[s] for s in u_stages],
    s=100,
    zorder=2,
    label="unique plans",
)
ax.scatter(
    [b / GB for b in p_bytes],
    p_iops,
    marker="*",
    s=300,
    c=[colors[s] for s in p_stages],
    edgecolors="black",
    linewidths=0.8,
    zorder=3,
    label="Pareto-optimal",
)
for s, b, iops in zip(u_stages, u_bytes, u_iops):
    ax.annotate(
        f" {s} stages", (b / GB, iops), fontsize=8, va="center"
    )
ax.set_xlabel("total bytes written (GB)")
ax.set_ylabel("max task IOps (in + out blocks)")
ax.set_title("2. Bytes vs IOps")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 3. Stages vs Bytes ───────────────────────────────────────────────────────
ax = axes[1, 0]
ax.scatter(
    u_stages,
    [b / GB for b in u_bytes],
    c=[colors[s] for s in u_stages],
    s=100,
    zorder=2,
)
ax.scatter(
    p_stages,
    [b / GB for b in p_bytes],
    marker="*",
    s=300,
    c=[colors[s] for s in p_stages],
    edgecolors="black",
    linewidths=0.8,
    zorder=3,
)
ax.set_xlabel("num_copy_ops (stages)")
ax.set_ylabel("total bytes written (GB)")
ax.set_title("3. Stages vs Bytes")
ax.set_xticks(stage_set)
ax.grid(True, alpha=0.3)

# ── 4. Stages vs IOps ────────────────────────────────────────────────────────
ax = axes[1, 1]
bar_width = 0.35
x = np.arange(len(u_stages))
bars_in = ax.bar(
    x - bar_width / 2, u_in, bar_width, label="max input blocks (fan-in)",
    color=[colors[s] for s in u_stages], alpha=0.7,
)
bars_out = ax.bar(
    x + bar_width / 2, u_out, bar_width, label="max output blocks (fan-out)",
    color=[colors[s] for s in u_stages], alpha=0.4, hatch="//",
)
for xi, s in enumerate(u_stages):
    if s in pareto_stages:
        ax.annotate("★", (xi, max(u_in[xi], u_out[xi]) + 5), ha="center", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{s} stages" for s in u_stages])
ax.set_ylabel("blocks per task")
ax.set_title("4. Fan-in vs Fan-out by stage count  (★ = Pareto-optimal)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = "rechunk-pareto-viz.svg"
plt.savefig(out)
print(f"Saved {out}")
plt.show()
