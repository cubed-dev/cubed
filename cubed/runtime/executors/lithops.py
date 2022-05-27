from functools import partial
from typing import Callable, Iterable

import networkx as nx
from lithops.executors import FunctionExecutor
from rechunker.types import ParallelPipelines, PipelineExecutor

from cubed.runtime.types import DagExecutor

# Lithops represents delayed execution tasks as functions that require
# a FunctionExecutor.
Task = Callable[[FunctionExecutor], None]


class LithopsPipelineExecutor(PipelineExecutor[Task]):
    """An execution engine based on Lithops."""

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Task:
        tasks = []
        for pipeline in pipelines:
            stage_tasks = []
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    stage_func = build_stage_mappable_func(stage, pipeline.config)
                    stage_tasks.append(stage_func)
                else:
                    stage_func = build_stage_func(stage, pipeline.config)
                    stage_tasks.append(stage_func)

            # Stages for a single pipeline must be executed in series
            tasks.append(partial(_execute_in_series, stage_tasks))

        # TODO: execute tasks for different specs in parallel
        return partial(_execute_in_series, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        with FunctionExecutor(**kwargs) as executor:
            plan(executor)


def build_stage_mappable_func(stage, config):
    def sf(mappable):
        return stage.function(mappable, config=config)

    def stage_func(lithops_function_executor):
        max_attempts = 3

        tagged_inputs = {k: v for (k, v) in enumerate(stage.mappable)}

        for _ in range(max_attempts):
            tagged_inputs_list = [[k, v] for (k, v) in tagged_inputs.items()]
            futures = lithops_function_executor.map(
                tagged_wrapper(sf), tagged_inputs_list
            )
            fs_done, _ = lithops_function_executor.wait(futures, throw_except=False)
            for f in fs_done:
                # remove successful tasks from tagged inputs and rerun others
                if f.success and not f.error:
                    tag = f.result()
                    del tagged_inputs[tag]

            if len(tagged_inputs) == 0:
                break

    return stage_func


def tagged_wrapper(func):
    def w(tagged_input, *args, **kwargs):
        tag, val = tagged_input
        func(val, *args, **kwargs)
        return tag

    return w


def build_stage_func(stage, config):
    def sf():
        return stage.function(config=config)

    def stage_func(lithops_function_executor):
        futures = lithops_function_executor.call_async(sf, ())
        lithops_function_executor.get_result(futures)

    return stage_func


def _execute_in_series(
    tasks: Iterable[Task], lithops_function_executor: FunctionExecutor
) -> None:
    for task in tasks:
        task(lithops_function_executor)


class LithopsDagExecutor(DagExecutor):

    # TODO: execute tasks for independent pipelines in parallel
    @staticmethod
    def execute_dag(dag, task_callback=None, **kwargs):
        if task_callback is not None:
            raise NotImplementedError("Task callback not supported")
        dag = dag.copy()
        with FunctionExecutor(**kwargs) as executor:
            for node in reversed(list(nx.topological_sort(dag))):
                pipeline = nx.get_node_attributes(dag, "pipeline").get(node, None)
                if pipeline is None:
                    continue

                for stage in pipeline.stages:
                    if stage.mappable is not None:
                        stage_func = build_stage_mappable_func(stage, pipeline.config)
                    else:
                        stage_func = build_stage_func(stage, pipeline.config)

                    # execute each stage in series
                    stage_func(executor)
