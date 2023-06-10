import copy
import logging
import time
from functools import partial
from typing import Callable, Iterable

from lithops.executors import FunctionExecutor
from lithops.wait import ALWAYS, ANY_COMPLETED
from six import reraise

from cubed.core.plan import visit_nodes
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import handle_callbacks
from cubed.vendor.rechunker.types import ParallelPipelines, PipelineExecutor

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
    timeout=None,
    retries=2,
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
    :param retries: The number of times to retry a failed task before raising an exception.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param return_stats: Whether to return lithops stats.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """
    return_when = ALWAYS if use_backups else ANY_COMPLETED

    inputs = list(map_iterdata)
    tasks = {}
    start_times = {}
    end_times = {}
    backups = {}
    pending = []

    # can't use functools.partial here as we get an error in lithops
    # also, lithops extra_args doesn't work for this case
    partial_map_function = lambda x: map_function(x, **kwargs)

    futures = lithops_function_executor.map(
        partial_map_function, inputs, timeout=timeout, include_modules=include_modules
    )
    tasks.update({k: v for (k, v) in zip(futures, inputs)})
    start_times.update({k: time.monotonic() for k in futures})
    pending.extend(futures)

    future_to_index = {f: i for i, f in enumerate(futures)}
    failure_counts = [0] * len(inputs)

    while pending:
        finished, pending = lithops_function_executor.wait(
            pending, throw_except=False, return_when=return_when, show_progressbar=False
        )
        failed = []
        for future in finished:
            if future.error:
                index = future_to_index[future]
                failure_counts[index] = failure_counts[index] + 1
                failure_count = failure_counts[index]

                if failure_count > retries:
                    # re-raise exception
                    # TODO: why does calling status not raise the exception?
                    future.status(throw_except=True)
                    reraise(*future._exception)
                failed.append(future)
            else:
                end_times[future] = time.monotonic()
                if return_stats:
                    yield future.result(), standardise_lithops_stats(future.stats)
                else:
                    yield future.result()

            # remove any backup task
            if use_backups:
                backup = backups.get(future, None)
                if backup:
                    if backup in pending:
                        pending.remove(backup)
                    del backups[future]
                    del backups[backup]

        if failed:
            # rerun and add to pending
            inputs_to_rerun = []
            input_indexes_to_rerun = []
            for fut in failed:
                index = future_to_index[fut]
                inputs_to_rerun.append(inputs[index])
                input_indexes_to_rerun.append(index)
                del future_to_index[fut]
            # TODO: de-duplicate code from above
            futures = lithops_function_executor.map(
                partial_map_function,
                inputs_to_rerun,
                timeout=timeout,
                include_modules=include_modules,
            )
            tasks.update({k: v for (k, v) in zip(futures, inputs_to_rerun)})
            start_times.update({k: time.monotonic() for k in futures})
            pending.extend(futures)

            future_to_index.update(
                {f: i for i, f in zip(input_indexes_to_rerun, futures)}
            )

        if use_backups:
            now = time.monotonic()
            for future in copy.copy(pending):
                if future not in backups and should_launch_backup(
                    future, now, start_times, end_times
                ):
                    input = tasks[future]
                    logger.info("Running backup task for %s", input)
                    futures = lithops_function_executor.map(
                        partial_map_function,
                        [input],
                        timeout=timeout,
                        include_modules=include_modules,
                    )
                    tasks.update({k: v for (k, v) in zip(futures, [input])})
                    start_times.update({k: time.monotonic() for k in futures})
                    pending.extend(futures)
                    backup = futures[0]  # TODO: launch multiple backups at once
                    backups[future] = backup
                    backups[backup] = future
            time.sleep(1)


def execute_dag(dag, callbacks=None, array_names=None, resume=None, **kwargs):
    use_backups = kwargs.pop("use_backups", False)
    with FunctionExecutor(**kwargs) as executor:
        for name, node in visit_nodes(dag, resume=resume):
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
                        stats["array_name"] = name
                        handle_callbacks(callbacks, stats)
                else:
                    raise NotImplementedError()


def standardise_lithops_stats(stats):
    return dict(
        task_create_tstamp=stats["host_job_create_tstamp"],
        function_start_tstamp=stats["worker_func_start_tstamp"],
        function_end_tstamp=stats["worker_func_end_tstamp"],
        task_result_tstamp=stats["host_status_done_tstamp"],
        peak_measured_mem_start=stats["worker_peak_memory_start"],
        peak_measured_mem_end=stats["worker_peak_memory_end"],
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
            stats["array_name"] = name
            handle_callbacks(callbacks, stats)

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

    def execute_dag(self, dag, callbacks=None, array_names=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        execute_dag(dag, callbacks=callbacks, array_names=array_names, **merged_kwargs)
