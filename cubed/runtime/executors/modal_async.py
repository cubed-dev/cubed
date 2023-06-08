import asyncio
import copy
import os
import time
from asyncio.exceptions import TimeoutError

import modal
import modal.aio
from modal.exception import ConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.core.plan import visit_nodes
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execute_with_stats, handle_callbacks

async_stub = modal.aio.AioStub("async-stub")

requirements_file = os.getenv("CUBED_MODAL_REQUIREMENTS_FILE")

if requirements_file:
    image = modal.Image.debian_slim().pip_install_from_requirements(requirements_file)
else:
    image = modal.Image.debian_slim().pip_install(
        [
            "fsspec",
            "mypy_extensions",  # for rechunker
            "networkx",
            "pytest-mock",  # TODO: only needed for tests
            "s3fs",
            "tenacity",
            "toolz",
            "zarr",
        ]
    )


@async_stub.function(
    image=image,
    secret=modal.Secret.from_name("my-aws-secret"),
    memory=2000,
    retries=2,
    is_generator=True,
)
async def async_run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    # note we can't use the execution_stat decorator since it doesn't work with modal decorators
    result, stats = execute_with_stats(func, input, config=config)
    yield result, stats


# We need map_unordered for the use_backups implementation
async def map_unordered(
    app_function, input, use_backups=False, return_stats=False, name=None, **kwargs
):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    :param app_function: The Modal function to map over the data.
    :param input: An iterable of input data.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param kwargs: Keyword arguments to pass to the function.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """
    task_create_tstamp = time.time()

    if not use_backups:
        async for result in app_function.map(input, kwargs=kwargs):
            if return_stats:
                result, stats = result
                if name is not None:
                    stats["array_name"] = name
                stats["task_create_tstamp"] = task_create_tstamp
                yield result, stats
            else:
                yield result
        return

    tasks = {asyncio.ensure_future(app_function.call(i, **kwargs)): i for i in input}
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
            # TODO: use exception groups in Python 3.11 to handle case of multiple task exceptions
            if task.exception():
                raise task.exception()
            end_times[task] = time.monotonic()
            if return_stats:
                result, stats = task.result()
                if name is not None:
                    stats["array_name"] = name
                stats["task_create_tstamp"] = task_create_tstamp
                yield result, stats
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
                    new_task = asyncio.ensure_future(app_function.call(i, **kwargs))
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
async def async_execute_dag(
    dag, callbacks=None, array_names=None, resume=None, **kwargs
):
    async with async_stub.run():
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    async for _, stats in map_unordered(
                        async_run_remotely,
                        list(stage.mappable),
                        return_stats=True,
                        name=name,
                        func=stage.function,
                        config=pipeline.config,
                        **kwargs,
                    ):
                        handle_callbacks(callbacks, stats)
                else:
                    raise NotImplementedError()


class AsyncModalDagExecutor(DagExecutor):
    """An execution engine that uses Modal's async API."""

    def execute_dag(self, dag, callbacks=None, array_names=None, resume=None, **kwargs):
        asyncio.run(
            async_execute_dag(
                dag,
                callbacks=callbacks,
                array_names=array_names,
                resume=resume,
                **kwargs,
            )
        )
