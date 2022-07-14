import time
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from cubed.core.array import Callback
from cubed.runtime.pipeline import already_computed


class HistoryCallback(Callback):
    def on_compute_start(self, arr):
        dag = arr.plan.optimize(arr.name)
        nodes = {n: d for (n, d) in dag.nodes(data=True)}
        plan = []
        for node in list(nx.topological_sort(dag)):
            if already_computed(nodes[node]):
                continue
            op_name = nodes[node]["op_name"]
            required_mem = nodes[node]["required_mem"]
            num_tasks = nodes[node]["num_tasks"]
            plan.append(
                dict(
                    array_name=node,
                    op_name=op_name,
                    required_mem=required_mem,
                    num_tasks=num_tasks,
                )
            )

        self.plan = plan
        self.stats = []

    def on_task_end(self, event):
        self.stats.append(asdict(event))

    def on_compute_end(self, arr):
        plan_df = pd.DataFrame(self.plan)
        stats_df = pd.DataFrame(self.stats)
        id = int(time.time())
        Path("history").mkdir(exist_ok=True)
        plan_df.to_csv(f"history/plan-{id}.csv", index=False)
        stats_df.to_csv(f"history/stats-{id}.csv", index=False)
