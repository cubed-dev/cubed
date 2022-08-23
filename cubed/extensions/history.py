import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cubed.core.array import Callback
from cubed.core.plan import visit_nodes


class HistoryCallback(Callback):
    def on_compute_start(self, dag):
        plan = []
        for name, node in visit_nodes(dag):
            plan.append(
                dict(
                    array_name=name,
                    op_name=node["op_name"],
                    required_mem=node["required_mem"],
                    num_tasks=node["num_tasks"],
                )
            )

        self.plan = plan
        self.stats = []

    def on_task_end(self, event):
        self.stats.append(asdict(event))

    def on_compute_end(self, dag):
        plan_df = pd.DataFrame(self.plan)
        stats_df = pd.DataFrame(self.stats)
        Path("history").mkdir(exist_ok=True)
        id = int(time.time())
        self.plan_df_path = Path(f"history/plan-{id}.csv")
        self.stats_df_path = Path(f"history/stats-{id}.csv")
        plan_df.to_csv(self.plan_df_path, index=False)
        stats_df.to_csv(self.stats_df_path, index=False)
