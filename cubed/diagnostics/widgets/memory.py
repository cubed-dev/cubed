import pathlib
import warnings

import anywidget
import pandas as pd
import traitlets
from IPython.display import display

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback


class MemoryWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "memory_widget.js"

    width = traitlets.Unicode("100%").tag(sync=True)
    height = traitlets.Unicode("300px").tag(sync=True)
    _num_tasks = traitlets.Int(0).tag(sync=True)
    _allowed_mem = traitlets.Int(0).tag(sync=True)
    _reserved_mem = traitlets.Int(0).tag(sync=True)
    _projected_mem_by_op = traitlets.List([]).tag(sync=True)  # type: ignore[var-annotated]


class LiveMemoryViewer(Callback):
    def __init__(self, widget=None, *, width="100%", height="300px"):
        self.widget = widget
        self.width = width
        self.height = height
        self.task_index = 0

    def on_compute_start(self, event):
        # find projected mem for each op
        self.num_tasks = event.plan.num_tasks
        self.projected_mem_by_op = []
        plan_ops = []
        task_index = 0
        for name, node in visit_nodes(event.dag):
            primitive_op = node["primitive_op"]
            plan_ops.append(
                dict(
                    name=name,
                    op_name=node["op_name"],
                    projected_mem=primitive_op.projected_mem,
                    allowed_mem=primitive_op.allowed_mem,
                    reserved_mem=primitive_op.reserved_mem,
                    num_tasks=primitive_op.num_tasks,
                )
            )
            # allowed and reserved are the same for all tasks
            self.allowed_mem = primitive_op.allowed_mem
            self.reserved_mem = primitive_op.reserved_mem
            self.projected_mem_by_op.append(
                dict(
                    name=name,
                    start_task_index=task_index,
                    end_task_index=task_index + primitive_op.num_tasks,
                    projected_mem=primitive_op.projected_mem // 1_000_000,
                )
            )
            task_index += primitive_op.num_tasks

        plan_ops_df = pd.DataFrame(plan_ops)
        self.projected_mem_map = plan_ops_df.set_index("name")[
            "projected_mem"
        ].to_dict()

        if self.widget is None:
            self.widget = MemoryWidget(
                width=self.width,
                height=self.height,
                _num_tasks=self.num_tasks,
                _allowed_mem=self.allowed_mem // 1_000_000,
                _reserved_mem=self.reserved_mem // 1_000_000,
                _projected_mem_by_op=self.projected_mem_by_op,
            )
            # show the widget in the current cell
            display(self.widget)
        else:
            self.widget._num_tasks = self.num_tasks
            self.widget._allowed_mem = self.allowed_mem // 1_000_000
            self.widget._reserved_mem = self.reserved_mem // 1_000_000
            self.widget._projected_mem_by_op = self.projected_mem_by_op

        self.widget.send({"type": "on_compute_start"})

    def on_task_end(self, event):
        if event.peak_measured_mem_end is None:
            warnings.warn(
                "Not displaying actual memory usage since it is not measured by the current executor",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.widget.send(
                {
                    "type": "on_task_end",
                    "name": event.name,
                    "task_index": self.task_index,
                    "actual_mem": event.peak_measured_mem_end // 1_000_000,
                    "projected_mem": self.projected_mem_map[event.name] // 1_000_000,
                }
            )

        # TODO: handle case where compute_arrays_in_parallel=False since task index
        # has to be incremented within the range of the tasks for the operation
        self.task_index += 1
