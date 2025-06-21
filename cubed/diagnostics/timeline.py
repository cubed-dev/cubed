from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cubed.runtime.types import Callback

sns.set_style("whitegrid")
matplotlib.use("Agg")


class TimelineVisualizationCallback(Callback):
    def __init__(self, format: Optional[str] = "svg") -> None:
        self.format = format

    def on_compute_start(self, event):
        self.events = []

    def on_task_end(self, event):
        self.events.append(asdict(event))

    def on_compute_end(self, event):
        events_df = pd.DataFrame(self.events)
        fig = generate_timeline(events_df)

        self.dst = Path(f"history/{event.compute_id}")
        self.dst.mkdir(parents=True, exist_ok=True)
        self.dst = self.dst / f"timeline.{self.format}"

        fig.savefig(self.dst)


def generate_timeline(events_df):
    events_df = events_df.sort_values(by=["task_create_tstamp", "name"], ascending=True)
    start_tstamp = events_df["task_create_tstamp"].min()
    total_calls = len(events_df)

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(total_calls)
    point_size = 7

    fields = [
        ("task create", events_df.task_create_tstamp - start_tstamp),
        ("function start", events_df.function_start_tstamp - start_tstamp),
        ("function end", events_df.function_end_tstamp - start_tstamp),
        ("task result", events_df.task_result_tstamp - start_tstamp),
    ]

    for f_i, (field_name, val) in enumerate(fields):
        ax.scatter(val, y, label=field_name, edgecolor="none", s=point_size, alpha=0.8)

    ax.set_xlabel("Execution time (sec)")
    ax.set_ylabel("Task number")

    ax.legend()

    return fig
