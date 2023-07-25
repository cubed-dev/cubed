from typing import Any, Callable, Optional, Sequence

from networkx import MultiDiGraph

from cubed.core.array import Callback, Spec, TaskEndEvent
from cubed.core.plan import visit_nodes
from cubed.primitive.types import CubedPipeline
from cubed.runtime.types import DagExecutor


def exec_stage_func(func: Callable[..., Any], *args, **kwargs):
    return func(*args, **kwargs)


class PythonDagExecutor(DagExecutor):
    """The default execution engine that runs tasks sequentially uses Python loops."""

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        array_names: Optional[Sequence[str]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        **kwargs,
    ) -> None:
        for name, node in visit_nodes(dag, resume=resume):
            pipeline: CubedPipeline = node["pipeline"]
            for m in pipeline.mappable:
                exec_stage_func(pipeline.function, m, config=pipeline.config)
                if callbacks is not None:
                    event = TaskEndEvent(array_name=name)
                    [callback.on_task_end(event) for callback in callbacks]
