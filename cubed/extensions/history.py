import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cubed.core.array import Callback
from cubed.core.plan import visit_nodes


class HistoryCallback(Callback):
    def on_compute_start(self, dag, resume):
        plan = []
        for name, node in visit_nodes(dag, resume):
            pipeline = node["pipeline"]
            plan.append(
                dict(
                    array_name=name,
                    op_name=node["op_name"],
                    projected_mem=pipeline.projected_mem,
                    reserved_mem=node["reserved_mem"],
                    num_tasks=pipeline.num_tasks,
                )
            )

        self.plan = plan
        self.events = []

    def on_task_end(self, event):
        self.events.append(asdict(event))

    def on_compute_end(self, dag):
        self.plan_df = pd.DataFrame(self.plan)
        self.events_df = pd.DataFrame(self.events)
        Path("history").mkdir(exist_ok=True)
        id = int(time.time())
        self.plan_df_path = Path(f"history/plan-{id}.csv")
        self.events_df_path = Path(f"history/events-{id}.csv")
        self.stats_df_path = Path(f"history/stats-{id}.csv")
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
            "array_name",
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
    df = events_df.groupby("array_name", as_index=False).agg(
        {
            "peak_measured_mem_start_mb": ["min", "mean", "max"],
            "peak_measured_mem_end_mb": ["max"],
            "peak_measured_mem_delta_mb": ["min", "mean", "max"],
        }
    )

    # flatten multi-index
    df.columns = ["_".join(a).rstrip("_") for a in df.columns.to_flat_index()]
    df = df.merge(plan_df, on="array_name")

    def projected_mem_utilization(row):
        return row["peak_measured_mem_end_mb_max"] / row["projected_mem_mb"]

    df["projected_mem_utilization"] = df.apply(
        lambda row: projected_mem_utilization(row), axis=1
    )
    df = df[
        [
            "array_name",
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
