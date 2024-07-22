import warnings
from collections import Counter

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback


class MemoryWarningCallback(Callback):
    def on_compute_start(self, event):
        # store ops keyed by name
        self.ops = {}
        for name, node in visit_nodes(event.dag, event.resume):
            primitive_op = node["primitive_op"]
            self.ops[name] = primitive_op

        # count number of times each op exceeds allowed mem
        self.counter = Counter()

    def on_task_end(self, event):
        allowed_mem = self.ops[event.name].allowed_mem
        if (
            event.peak_measured_mem_end is not None
            and event.peak_measured_mem_end > allowed_mem
        ):
            self.counter.update({event.name: 1})

    def on_compute_end(self, event):
        if sum(self.counter.values()) > 0:
            exceeded = [
                f"{k} ({v}/{self.ops[k].num_tasks})" for k, v in self.counter.items()
            ]
            warnings.warn(
                f"Peak memory usage exceeded allowed_mem when running tasks: {', '.join(exceeded)}",
                UserWarning,
            )
