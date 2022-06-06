from functools import partial
from typing import Callable, Iterable

import networkx as nx
from lithops.executors import FunctionExecutor
from lithops.wait import ANY_COMPLETED
from rechunker.types import ParallelPipelines, PipelineExecutor

from cubed.runtime.pipeline import already_computed
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

        attempt = 0

        tagged_inputs = {k: v for (k, v) in enumerate(stage.mappable)}
        futures = []
        tagged_futures = {}
        pending = []

        tagged_inputs_list = [[k, v] for (k, v) in tagged_inputs.items()]
        futures.extend(
            lithops_function_executor.map(
                tagged_wrapper(sf), tagged_inputs_list, include_modules=["cubed"]
            )
        )
        tagged_futures.update({k: v for (k, v) in zip(futures, tagged_inputs_list)})
        pending.extend(futures)
        attempt += 1

        while pending:
            throw_except = attempt == max_attempts - 1
            finished, pending = lithops_function_executor.wait(
                futures, throw_except=throw_except, return_when=ANY_COMPLETED
            )

            errored = []
            for future in finished:
                if future.error:
                    errored.append(future)
                futures.remove(future)
            if errored:
                # rerun and add to pending
                tagged_inputs_list = [
                    [k, v] for (fut, (k, v)) in tagged_futures.items() if fut in errored
                ]
                # TODO: de-duplicate code from above
                new_futures = lithops_function_executor.map(
                    tagged_wrapper(sf),
                    tagged_inputs_list,
                    include_modules=["cubed"],
                )
                futures.extend(new_futures)
                tagged_futures.update(
                    {k: v for (k, v) in zip(futures, tagged_inputs_list)}
                )
                pending.append(new_futures)
                attempt += 1

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
    def execute_dag(dag, callbacks=None, **kwargs):
        if callbacks is not None:
            raise NotImplementedError("Callbacks not supported")
        with FunctionExecutor(**kwargs) as executor:
            nodes = {n: d for (n, d) in dag.nodes(data=True)}
            for node in reversed(list(nx.topological_sort(dag))):
                if already_computed(nodes[node]):
                    continue
                pipeline = nodes[node]["pipeline"]

                for stage in pipeline.stages:
                    if stage.mappable is not None:
                        stage_func = build_stage_mappable_func(stage, pipeline.config)
                    else:
                        stage_func = build_stage_func(stage, pipeline.config)

                    # execute each stage in series
                    stage_func(executor)
