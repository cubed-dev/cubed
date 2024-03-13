from typing import Optional, Sequence

from networkx import MultiDiGraph

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback, CubedPipeline, DagExecutor, TaskEndEvent
from cubed.runtime.utils import handle_operation_start_callbacks
from cubed.spec import Spec


def exec_stage_func(input, func=None, config=None, name=None, compute_id=None):
    return func(input, config=config)


class PythonDagExecutor(DagExecutor):
    """The default execution engine that runs tasks sequentially uses Python loops."""

    @property
    def name(self) -> str:
        return "single-threaded"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        for name, node in visit_nodes(dag, resume=resume):
            handle_operation_start_callbacks(callbacks, name)
            pipeline: CubedPipeline = node["pipeline"]
            for m in pipeline.mappable:
                exec_stage_func(
                    m,
                    pipeline.function,
                    config=pipeline.config,
                    name=name,
                    compute_id=compute_id,
                )
                if callbacks is not None:
                    event = TaskEndEvent(name=name)
                    [callback.on_task_end(event) for callback in callbacks]
