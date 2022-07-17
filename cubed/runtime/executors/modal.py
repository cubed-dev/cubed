import asyncio
import copy
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

from cubed.core.array import TaskEndEvent
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import DagExecutor
from cubed.utils import peak_memory

stub = modal.Stub()
async_stub = modal.aio.AioStub()

image = modal.DebianSlim().pip_install(
    [
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


@stub.function(image=image, secret=modal.ref("my-aws-secret"))
def run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    # we can't return None, as Modal map won't work in that case
    return func(input, config=config) or 1


@async_stub.function(image=image, secret=modal.ref("my-aws-secret"))
async def async_run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    peak_memory_start = peak_memory()
    function_start_tstamp = time.time()
    result = func(input, config=config)
    function_end_tstamp = time.time()
    peak_memory_end = peak_memory()
    return (
        result,
        function_start_tstamp,
        function_end_tstamp,
        peak_memory_start,
        peak_memory_end,
    )


async def map_unordered(
    app_function, input, max_failures=3, use_backups=False, return_stats=False, **kwargs
):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    :param app_function: The Modal function to map over the data.
    :param input: An iterable of input data.
    :param max_failures: The number of task failures to allow before raising an exception.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param return_stats: Whether to return execution stats.
    :param kwargs: Keyword arguments to pass to the function.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """
    failures = 0
    tasks = {asyncio.ensure_future(app_function(i, **kwargs)): i for i in input}
    pending = set(tasks.keys())
    t = time.monotonic()
    start_times = {f: t for f in pending}
    end_times = {}
    backups = {}

    event_t = time.time()
    event_start_times = {f: event_t for f in pending}
    event_end_times = {}

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
                if return_stats:
                    event_end_times[task] = time.time()
                    (
                        res,
                        function_start_tstamp,
                        function_end_tstamp,
                        peak_memory_start,
                        peak_memory_end,
                    ) = task.result()
                    task_stats = dict(
                        task_create_tstamp=event_start_times[task],
                        function_start_tstamp=function_start_tstamp,
                        function_end_tstamp=function_end_tstamp,
                        task_result_tstamp=event_end_times[task],
                        peak_memory_start=peak_memory_start,
                        peak_memory_end=peak_memory_end,
                    )
                    yield res, task_stats
                else:
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
                if task not in backups and should_launch_backup(
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


def execute_dag(dag, callbacks=None, **kwargs):
    max_attempts = 3
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(TimeoutError),
            stop=stop_after_attempt(max_attempts),
        ):
            with attempt:
                with stub.run():

                    nodes = {n: d for (n, d) in dag.nodes(data=True)}
                    for node in list(nx.topological_sort(dag)):
                        if already_computed(nodes[node]):
                            continue
                        pipeline = nodes[node]["pipeline"]

                        for stage in pipeline.stages:
                            if stage.mappable is not None:
                                # print(f"about to run remotely on {stage.mappable}")
                                for _ in run_remotely.map(
                                    list(stage.mappable),
                                    kwargs=dict(
                                        func=stage.function, config=pipeline.config
                                    ),
                                ):
                                    if callbacks is not None:
                                        event = TaskEndEvent(array_name=node)
                                        [
                                            callback.on_task_end(event)
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

                async with async_stub.run():
                    nodes = {n: d for (n, d) in dag.nodes(data=True)}
                    for node in list(nx.topological_sort(dag)):
                        if already_computed(nodes[node]):
                            continue
                        pipeline = nodes[node]["pipeline"]

                        for stage in pipeline.stages:
                            if stage.mappable is not None:
                                # print(f"about to run remotely on {stage.mappable}")
                                async for _, stats in map_unordered(
                                    async_run_remotely,
                                    list(stage.mappable),
                                    return_stats=True,
                                    func=stage.function,
                                    config=pipeline.config,
                                ):
                                    if callbacks is not None:
                                        event = TaskEndEvent(array_name=node, **stats)
                                        [
                                            callback.on_task_end(event)
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
