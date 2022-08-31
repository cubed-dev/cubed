import copy
import logging
import time
from functools import partial
from typing import Callable, Iterable

from lithops.executors import FunctionExecutor
from lithops.wait import ALWAYS, ANY_COMPLETED
from rechunker.types import ParallelPipelines, PipelineExecutor
from six import reraise

from cubed.core.array import TaskEndEvent
from cubed.core.plan import visit_nodes
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.types import DagExecutor

logger = logging.getLogger(__name__)

# Lithops represents delayed execution tasks as functions that require
# a FunctionExecutor.
Task = Callable[[FunctionExecutor], None]


class LithopsPipelineExecutor(PipelineExecutor[Task]):
    """An execution engine based on Lithops."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

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

        return partial(_execute_in_series, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        with FunctionExecutor(**merged_kwargs) as executor:
            plan(executor)


def run_func(input, func=None, config=None, name=None):
    result = func(input, config=config)
    return result


def map_unordered(
    lithops_function_executor,
    map_function,
    map_iterdata,
    include_modules=[],
    max_failures=3,
    use_backups=False,
    return_stats=False,
    **kwargs,
):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    A generalisation of Lithops `map`, with retries, and relaxed return ordering.

    :param lithops_function_executor: The Lithops function executor to use.
    :param map_function: The function to map over the data.
    :param map_iterdata: An iterable of input data.
    :param include_modules: Modules to include.
    :param max_failures: The number of task failures to allow before raising an exception.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param return_stats: Whether to return lithops stats.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """
    failures = 0
    return_when = ALWAYS if use_backups else ANY_COMPLETED

    inputs = map_iterdata
    tasks = {}
    start_times = {}
    end_times = {}
    backups = {}
    pending = []

    # can't use functools.partial here as we get an error in lithops
    # also, lithops extra_args doesn't work for this case
    partial_map_function = lambda x: map_function(x, **kwargs)

    futures = lithops_function_executor.map(
        partial_map_function, inputs, include_modules=include_modules
    )
    tasks.update({k: v for (k, v) in zip(futures, inputs)})
    start_times.update({k: time.monotonic() for k in futures})
    pending.extend(futures)

    while pending:
        finished, pending = lithops_function_executor.wait(
            pending, throw_except=False, return_when=return_when, show_progressbar=False
        )

        failed = []
        for future in finished:
            if future.error:
                failures += 1
                if failures > max_failures:
                    # re-raise exception
                    # TODO: why does calling status not raise the exception?
                    future.status(throw_except=True)
                    reraise(*future._exception)
                failed.append(future)
            else:
                end_times[future] = time.monotonic()
                if return_stats:
                    yield future.result(), future.stats
                else:
                    yield future.result()

            if use_backups:
                # remove backups
                backup = backups.get(future, None)
                if backup:
                    if backup in pending:
                        pending.remove(backup)
                    del backups[future]
                    del backups[backup]

        if failed:
            # rerun and add to pending
            inputs = [v for (fut, v) in tasks.items() if fut in failed]
            # TODO: de-duplicate code from above
            futures = lithops_function_executor.map(
                partial_map_function,
                inputs,
                include_modules=include_modules,
            )
            tasks.update({k: v for (k, v) in zip(futures, inputs)})
            start_times.update({k: time.monotonic() for k in futures})
            pending.extend(futures)

        if use_backups:
            now = time.monotonic()
            for future in copy.copy(pending):
                if future not in backups and should_launch_backup(
                    future, now, start_times, end_times
                ):
                    inputs = [v for (fut, v) in tasks.items() if fut == future]
                    logger.info("Running backup task for %s", inputs)
                    futures = lithops_function_executor.map(
                        partial_map_function,
                        inputs,
                        include_modules=include_modules,
                    )
                    tasks.update({k: v for (k, v) in zip(futures, inputs)})
                    start_times.update({k: time.monotonic() for k in futures})
                    pending.extend(futures)
                    pending.remove(future)  # throw away slow one
                    backup = futures[0]  # TODO: launch multiple backups at once
                    backups[future] = backup
                    backups[backup] = future
            time.sleep(1)


def execute_dag(dag, callbacks=None, **kwargs):
    use_backups = kwargs.pop("use_backups", False)
    with FunctionExecutor(**kwargs) as executor:
        for name, node in visit_nodes(dag):
            pipeline = node["pipeline"]
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    for _, stats in map_unordered(
                        executor,
                        run_func,
                        list(stage.mappable),
                        func=stage.function,
                        config=pipeline.config,
                        name=name,
                        use_backups=use_backups,
                        return_stats=True,
                    ):
                        if callbacks is not None:
                            event = lithops_stats_to_task_end_event(name, stats)
                            [callback.on_task_end(event) for callback in callbacks]
                else:
                    raise NotImplementedError()


def lithops_stats_to_task_end_event(name, stats):
    return TaskEndEvent(
        array_name=name,
        task_create_tstamp=stats["host_job_create_tstamp"],
        function_start_tstamp=stats["worker_func_start_tstamp"],
        function_end_tstamp=stats["worker_func_end_tstamp"],
        task_result_tstamp=stats["host_status_done_tstamp"],
        peak_memory_start=stats["worker_peak_memory_start"],
        peak_memory_end=stats["worker_peak_memory_end"],
    )


def build_stage_mappable_func(
    stage, config, name=None, callbacks=None, use_backups=False
):
    def sf(mappable):
        return stage.function(mappable, config=config)

    def stage_func(lithops_function_executor):
        for _, stats in map_unordered(
            lithops_function_executor,
            sf,
            list(stage.mappable),
            use_backups=use_backups,
            return_stats=True,
        ):
            if callbacks is not None:
                event = lithops_stats_to_task_end_event(name, stats)
                [callback.on_task_end(event) for callback in callbacks]

    return stage_func


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
    """An execution engine that uses Lithops."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute_dag(self, dag, callbacks=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        execute_dag(dag, callbacks=callbacks, **merged_kwargs)
