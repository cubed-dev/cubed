from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback

matplotlib.use("Agg")


class MemoryVisualizationCallback(Callback):
    def __init__(self, format: Optional[str] = "svg") -> None:
        self.format = format

    def on_compute_start(self, event):
        plan = []
        for name, node in visit_nodes(event.dag):
            primitive_op = node["primitive_op"]
            plan.append(
                dict(
                    name=name,
                    op_name=node["op_name"],
                    projected_mem=primitive_op.projected_mem,
                    allowed_mem=primitive_op.allowed_mem,
                    reserved_mem=primitive_op.reserved_mem,
                    num_tasks=primitive_op.num_tasks,
                )
            )

        self.plan = plan
        self.events = []

    def on_task_end(self, event):
        self.events.append(asdict(event))

    def on_compute_end(self, event):
        events_df = pd.DataFrame(self.events)
        plan_df = pd.DataFrame(self.plan)
        fig = generate_mem_usage(events_df, plan_df)

        self.dst = Path(f"history/{event.compute_id}")
        self.dst.mkdir(parents=True, exist_ok=True)
        self.dst = self.dst / f"memory.{self.format}"

        fig.savefig(self.dst)


def generate_mem_usage(events_df, plan_df):
    # colours match those in https://cubed-dev.github.io/cubed/user-guide/memory.html

    events_df = events_df.sort_values(by=["task_create_tstamp", "name"], ascending=True)
    projected_mem_map = plan_df.set_index("name")["projected_mem"].to_dict()

    tstamp = events_df["task_result_tstamp"].astype("timedelta64[s]")
    events_df["time"] = (tstamp - tstamp.min()).astype(int)
    events_df["actual usage"] = events_df["peak_measured_mem_end"] / 1_000_000
    events_df["projected_mem"] = events_df.name.map(projected_mem_map) / 1_000_000

    fig, ax = plt.subplots(figsize=(8, 6))

    events_df.plot(
        kind="area", y="actual usage", ax=ax, use_index=True, color="#9fc5e8"
    )

    allowed_mem = plan_df["allowed_mem"].max() / 1_000_000
    ax.axhline(allowed_mem, label="allowed", color="#e06666", linestyle="--")

    reserved_mem = plan_df["reserved_mem"].max() / 1_000_000
    ax.axhline(
        reserved_mem,
        label="reserved",
        color="#f6b26b",
        linestyle="--",
    )

    peak_measured_mem = events_df["peak_measured_mem_end"].max() / 1_000_000
    ax.axhline(peak_measured_mem, label="max actual usage", color="#6fa8dc")

    events_df.plot(
        kind="line",
        y="projected_mem",
        ax=ax,
        use_index=True,
        label="projected",
        color="#93c47d",
        linestyle="--",
    )

    ax.set_xlabel("Task number")
    ax.set_ylim(top=allowed_mem + 100)
    ax.set_ylabel("Task memory (MB)")
    ax.legend()

    return fig
