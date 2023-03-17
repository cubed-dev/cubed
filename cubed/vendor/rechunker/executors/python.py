from typing import Callable

from ..types import ParallelPipelines, PipelineExecutor

# PythonExecutor represents delayed execution tasks as functions that require
# no arguments.
Task = Callable[[], None]


class PythonPipelineExecutor(PipelineExecutor[Task]):
    """An execution engine based on Python loops.

    Execution plans for PythonExecutor are functions that accept no arguments.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Task:
        def plan():
            for pipeline in pipelines:
                for stage in pipeline.stages:
                    if stage.mappable is not None:
                        for m in stage.mappable:
                            stage.function(m, config=pipeline.config)
                    else:
                        stage.function(config=pipeline.config)

        return plan

    def execute_plan(self, plan: Task, **kwargs):
        plan()
