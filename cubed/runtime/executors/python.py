import networkx as nx
from tenacity import retry, stop_after_attempt

from cubed.core.array import TaskEndEvent
from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import DagExecutor


@retry(stop=stop_after_attempt(3))
def exec_stage_func(func, *args, **kwargs):
    return func(*args, **kwargs)


class PythonDagExecutor(DagExecutor):
    def execute_dag(self, dag, callbacks=None, **kwargs):
        nodes = {n: d for (n, d) in dag.nodes(data=True)}
        for node in list(nx.topological_sort(dag)):
            if already_computed(nodes[node]):
                continue
            pipeline = nodes[node]["pipeline"]
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    for m in stage.mappable:
                        exec_stage_func(stage.function, m, config=pipeline.config)
                        if callbacks is not None:
                            event = TaskEndEvent(array_name=node)
                            [callback.on_task_end(event) for callback in callbacks]
                else:
                    exec_stage_func(stage.function, config=pipeline.config)
                    if callbacks is not None:
                        event = TaskEndEvent(array_name=node)
                        [callback.on_task_end(event) for callback in callbacks]
