from tenacity import retry, stop_after_attempt

from cubed.core.array import TaskEndEvent
from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor


@retry(stop=stop_after_attempt(3))
def exec_stage_func(func, *args, **kwargs):
    return func(*args, **kwargs)


class PythonDagExecutor(DagExecutor):
    """An execution engine that uses Python loops."""

    def execute_dag(self, dag, callbacks=None, **kwargs):
        for name, node in visit_nodes(dag):
            pipeline = node["pipeline"]
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    for m in stage.mappable:
                        exec_stage_func(stage.function, m, config=pipeline.config)
                        if callbacks is not None:
                            event = TaskEndEvent(array_name=name)
                            [callback.on_task_end(event) for callback in callbacks]
                else:
                    exec_stage_func(stage.function, config=pipeline.config)
                    if callbacks is not None:
                        event = TaskEndEvent(array_name=name)
                        [callback.on_task_end(event) for callback in callbacks]
