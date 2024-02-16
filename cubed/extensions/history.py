from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback


class HistoryCallback(Callback):
    def on_compute_start(self, event):
        plan = []
        for name, node in visit_nodes(event.dag, event.resume):
            primitive_op = node["primitive_op"]
            plan.append(
                dict(
                    name=name,
                    op_name=node["op_name"],
                    projected_mem=primitive_op.projected_mem,
                    reserved_mem=primitive_op.reserved_mem,
                    num_tasks=primitive_op.num_tasks,
                )
            )

        self.plan = plan
        self.events = []

    def on_task_end(self, event):
        self.events.append(asdict(event))

    def on_compute_end(self, event):
        self.plan_df = pd.DataFrame(self.plan)
        self.events_df = pd.DataFrame(self.events)
        history_path = Path(f"history/{event.compute_id}")
        history_path.mkdir(parents=True, exist_ok=True)
        self.plan_df_path = history_path / "plan.csv"
        self.events_df_path = history_path / "events.csv"
        self.stats_df_path = history_path / "stats.csv"
        self.plan_df.to_csv(self.plan_df_path, index=False)
        self.events_df.to_csv(self.events_df_path, index=False)

        self.stats_df = analyze(self.plan_df, self.events_df)
        self.stats_df.to_csv(self.stats_df_path, index=False)


def analyze(plan_df, events_df):
    # convert memory to MB
    plan_df["projected_mem_mb"] = plan_df["projected_mem"] / 1_000_000
    plan_df["reserved_mem_mb"] = plan_df["reserved_mem"] / 1_000_000
    plan_df = plan_df[
        [
            "name",
            "op_name",
            "projected_mem_mb",
            "reserved_mem_mb",
            "num_tasks",
        ]
    ]
    events_df["peak_measured_mem_start_mb"] = (
        events_df["peak_measured_mem_start"] / 1_000_000
    )
    events_df["peak_measured_mem_end_mb"] = (
        events_df["peak_measured_mem_end"] / 1_000_000
    )
    events_df["peak_measured_mem_delta_mb"] = (
        events_df["peak_measured_mem_end_mb"] - events_df["peak_measured_mem_start_mb"]
    )

    # find per-array stats
    df = events_df.groupby("name", as_index=False).agg(
        {
            "peak_measured_mem_start_mb": ["min", "mean", "max"],
            "peak_measured_mem_end_mb": ["max"],
            "peak_measured_mem_delta_mb": ["min", "mean", "max"],
        }
    )

    # flatten multi-index
    df.columns = ["_".join(a).rstrip("_") for a in df.columns.to_flat_index()]
    df = df.merge(plan_df, on="name")

    def projected_mem_utilization(row):
        return row["peak_measured_mem_end_mb_max"] / row["projected_mem_mb"]

    df["projected_mem_utilization"] = df.apply(
        lambda row: projected_mem_utilization(row), axis=1
    )
    df = df[
        [
            "name",
            "op_name",
            "num_tasks",
            "peak_measured_mem_start_mb_max",
            "peak_measured_mem_end_mb_max",
            "peak_measured_mem_delta_mb_max",
            "projected_mem_mb",
            "reserved_mem_mb",
            "projected_mem_utilization",
        ]
    ]

    return df
