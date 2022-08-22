import asyncio
import copy
import time
from asyncio.exceptions import TimeoutError

import modal
import modal.aio
from modal.exception import ConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.core.array import TaskEndEvent
from cubed.core.plan import visit_nodes
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.types import DagExecutor
from cubed.utils import peak_memory

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


@async_stub.generator(image=image, secret=modal.ref("my-aws-secret"), retries=2)
async def async_run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    peak_memory_start = peak_memory()
    function_start_tstamp = time.time()
    result = func(input, config=config)
    function_end_tstamp = time.time()
    peak_memory_end = peak_memory()
    yield (
        result,
        function_start_tstamp,
        function_end_tstamp,
        peak_memory_start,
        peak_memory_end,
    )


# We need map_unordered for the use_backups implementation
async def map_unordered(
    app_function, input, max_failures=3, use_backups=False, **kwargs
):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    :param app_function: The Modal function to map over the data.
    :param input: An iterable of input data.
    :param max_failures: The number of task failures to allow before raising an exception.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param kwargs: Keyword arguments to pass to the function.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """
    if not use_backups:
        async for result in app_function.map(input, kwargs=kwargs):
            yield result
        return

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
                raise task.exception()
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


# This just retries the initial connection attempt, not the function calls
@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
)
async def async_execute_dag(dag, callbacks=None, **kwargs):
    async with async_stub.run():
        for name, node in visit_nodes(dag):
            pipeline = node["pipeline"]

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    task_create_tstamp = time.time()
                    async for result in map_unordered(
                        async_run_remotely,
                        list(stage.mappable),
                        func=stage.function,
                        config=pipeline.config,
                        **kwargs,
                    ):
                        handle_callbacks(callbacks, name, result, task_create_tstamp)
                else:
                    raise NotImplementedError()


def handle_callbacks(callbacks, array_name, result, task_create_tstamp):
    if callbacks is not None:
        task_result_tstamp = time.time()
        (
            res,
            function_start_tstamp,
            function_end_tstamp,
            peak_memory_start,
            peak_memory_end,
        ) = result
        task_stats = dict(
            task_create_tstamp=task_create_tstamp,
            function_start_tstamp=function_start_tstamp,
            function_end_tstamp=function_end_tstamp,
            task_result_tstamp=task_result_tstamp,
            peak_memory_start=peak_memory_start,
            peak_memory_end=peak_memory_end,
        )
        event = TaskEndEvent(array_name=array_name, **task_stats)
        [callback.on_task_end(event) for callback in callbacks]


class AsyncModalDagExecutor(DagExecutor):
    """An execution engine that uses Modal's async API."""

    def execute_dag(self, dag, callbacks=None, **kwargs):
        asyncio.run(async_execute_dag(dag, callbacks=callbacks, **kwargs))
