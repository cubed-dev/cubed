import os
import time
from dataclasses import asdict
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pylab
import seaborn as sns

from cubed.runtime.types import Callback

sns.set_style("whitegrid")
pylab.switch_backend("Agg")


class TimelineVisualizationCallback(Callback):
    def __init__(self, format: Optional[str] = None) -> None:
        self.format = format

    def on_compute_start(self, event):
        self.start_tstamp = time.time()
        self.stats = []

    def on_task_end(self, event):
        self.stats.append(asdict(event))

    def on_compute_end(self, event):
        end_tstamp = time.time()
        dst = f"history/{event.compute_id}"
        format = self.format
        create_timeline(self.stats, self.start_tstamp, end_tstamp, dst, format)


# copy of lithops function of the same name, and modified for different field names
def create_timeline(stats, start_tstamp, end_tstamp, dst=None, format=None):
    stats_df = pd.DataFrame(stats)

    stats_df = stats_df.sort_values(by=["task_create_tstamp", "name"], ascending=True)

    total_calls = len(stats_df)

    palette = sns.color_palette("deep", 6)

    fig = pylab.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    y = np.arange(total_calls)
    point_size = 10

    fields = [
        ("task create", stats_df.task_create_tstamp - start_tstamp),
        ("function start", stats_df.function_start_tstamp - start_tstamp),
        ("function end", stats_df.function_end_tstamp - start_tstamp),
        ("task result", stats_df.task_result_tstamp - start_tstamp),
    ]

    patches = []
    for f_i, (field_name, val) in enumerate(fields):
        ax.scatter(val, y, c=[palette[f_i]], edgecolor="none", s=point_size, alpha=0.8)
        patches.append(mpatches.Patch(color=palette[f_i], label=field_name))

    ax.set_xlabel("Execution Time (sec)")
    ax.set_ylabel("Function Call")

    legend = pylab.legend(handles=patches, loc="upper right", frameon=True)
    legend.get_frame().set_facecolor("#FFFFFF")

    yplot_step = int(np.max([1, total_calls / 20]))
    y_ticks = np.arange(total_calls // yplot_step + 2) * yplot_step
    ax.set_yticks(y_ticks)
    ax.set_ylim(-0.02 * total_calls, total_calls * 1.02)
    for y in y_ticks:
        ax.axhline(y, c="k", alpha=0.1, linewidth=1)

    max_seconds = np.max(end_tstamp - start_tstamp) * 1.25
    xplot_step = max(int(max_seconds / 8), 1)
    x_ticks = np.arange(max_seconds // xplot_step + 2) * xplot_step
    ax.set_xlim(0, max_seconds)

    ax.set_xticks(x_ticks)
    for x in x_ticks:
        ax.axvline(x, c="k", alpha=0.2, linewidth=0.8)

    ax.grid(False)
    fig.tight_layout()

    if format is None:
        format = "svg"

    if dst is None:
        os.makedirs("plots", exist_ok=True)
        dst = os.path.join(
            os.getcwd(), "plots", "{}_{}".format(int(time.time()), f"timeline.{format}")
        )
    else:
        dst = os.path.expanduser(dst) if "~" in dst else dst
        dst = "{}/{}".format(os.path.realpath(dst), f"timeline.{format}")

    fig.savefig(dst)
