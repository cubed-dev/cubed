import asyncio
import copy
import math
import time
from asyncio.exceptions import TimeoutError

import modal
import modal.aio
import networkx as nx
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import DagExecutor

app = modal.App()
async_app = modal.aio.AioApp()

image = modal.DebianSlim(
    python_packages=[
        "dask[array]",
        "fsspec",
        "networkx",
        "pytest-mock",  # TODO: only needed for tests
        "rechunker",
        "s3fs",
        "tenacity",
        "zarr",
    ]
)


@app.function(image=image, secret=modal.ref("my-aws-secret"))
def run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    # we can't return None, as Modal map won't work in that case
    return func(input, config=config) or 1


@async_app.function(image=image, secret=modal.ref("my-aws-secret"))
async def async_run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    return func(input, config=config)


async def map_as_completed(
    app_function, input, max_failures=3, use_backups=True, **kwargs
):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    :param app_function: The Modal function to map over the data.
    :param input: An iterable of input data.
    :param max_failures: The number of task failures to allow before raising an exception.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param kwargs: Keyword arguments to pass to the function.

    :return: Function values as they are completed, not necessarily in the input order.
    """
    failures = 0
    tasks = {asyncio.ensure_future(app_function(i, **kwargs)): i for i in input}
    pending = set(tasks.keys())
    t = time.monotonic()
    start_times = {f: t for f in pending}
    end_times = {}
    backups = {}

    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED, timeout=2
        )
        # print("finished", finished)
        # print("pending", pending)

        for task in finished:
            if task.exception():
                failures += 1
                if failures > max_failures:
                    raise task.exception()
                # retry failed task
                i = tasks[task]
                new_task = asyncio.ensure_future(app_function(i, **kwargs))
                tasks[new_task] = i
                start_times[new_task] = time.monotonic()
                pending.add(new_task)
            else:
                end_times[task] = time.monotonic()
                yield task.result()

            # remove any backup task
            if use_backups:
                backup = backups.get(task, None)
                if backup:
                    pending.remove(backup)
                    del backups[task]
                    del backups[backup]

        if use_backups:
            now = time.monotonic()
            for task in copy.copy(pending):
                if task not in backups and launch_backup(
                    task, now, start_times, end_times
                ):
                    # launch backup task
                    i = tasks[task]
                    new_task = asyncio.ensure_future(app_function(i, **kwargs))
                    tasks[new_task] = i
                    start_times[new_task] = time.monotonic()
                    pending.add(new_task)
                    backups[task] = new_task
                    backups[new_task] = task


def launch_backup(
    task,
    now,
    start_times,
    end_times,
    min_tasks=10,
    min_completed_fraction=0.5,
    slow_factor=3.0,
):
    """
    Determine whether to launch a backup task.

    Backup tasks are only launched if there are at least `min_tasks` being run, and `min_completed_fraction` of tasks have completed.
    If both those criteria have been met, then a backup task is launched if the duration of the current task is at least
    `slow_factor` times slower than the `min_completed_fraction` percentile task duration.
    """
    if len(start_times) < min_tasks:
        return False
    n = math.ceil(len(start_times) * min_completed_fraction) - 1
    if len(end_times) <= n:
        return False
    completed_durations = sorted(
        [end_times[task] - start_times[task] for task in end_times]
    )
    duration = now - start_times[task]
    return duration > completed_durations[n] * slow_factor


def execute_dag(dag, callbacks=None, **kwargs):
    max_attempts = 3
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(TimeoutError),
            stop=stop_after_attempt(max_attempts),
        ):
            with attempt:
                with app.run():

                    nodes = {n: d for (n, d) in dag.nodes(data=True)}
                    for node in reversed(list(nx.topological_sort(dag))):
                        if already_computed(nodes[node]):
                            continue
                        pipeline = nodes[node]["pipeline"]

                        for stage in pipeline.stages:
                            if stage.mappable is not None:
                                # print(f"about to run remotely on {stage.mappable}")
                                for _ in run_remotely.map(
                                    list(stage.mappable),
                                    window=1,
                                    kwargs=dict(
                                        func=stage.function, config=pipeline.config
                                    ),
                                ):
                                    if callbacks is not None:
                                        [
                                            callback.on_task_end()
                                            for callback in callbacks
                                        ]
                            else:
                                raise NotImplementedError()
    except RetryError:
        pass


async def async_execute_dag(dag, callbacks=None, **kwargs):
    max_attempts = 3
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(TimeoutError),
            stop=stop_after_attempt(max_attempts),
        ):
            with attempt:

                async with async_app.run():
                    nodes = {n: d for (n, d) in dag.nodes(data=True)}
                    for node in reversed(list(nx.topological_sort(dag))):
                        if already_computed(nodes[node]):
                            continue
                        pipeline = nodes[node]["pipeline"]

                        for stage in pipeline.stages:
                            if stage.mappable is not None:
                                # print(f"about to run remotely on {stage.mappable}")
                                async for _ in map_as_completed(
                                    async_run_remotely,
                                    list(stage.mappable),
                                    func=stage.function,
                                    config=pipeline.config,
                                ):
                                    if callbacks is not None:
                                        [
                                            callback.on_task_end()
                                            for callback in callbacks
                                        ]
                            else:
                                raise NotImplementedError()
    except RetryError:
        pass


class ModalDagExecutor(DagExecutor):
    # TODO: execute tasks for independent pipelines in parallel
    def execute_dag(self, dag, callbacks=None, **kwargs):
        execute_dag(dag, callbacks=callbacks)


class AsyncModalDagExecutor(DagExecutor):
    def execute_dag(self, dag, callbacks=None, **kwargs):
        asyncio.run(async_execute_dag(dag, callbacks=callbacks))
