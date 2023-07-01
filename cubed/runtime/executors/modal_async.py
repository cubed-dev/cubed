import asyncio
import copy
import time
from asyncio.exceptions import TimeoutError

from modal.exception import ConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.core.plan import visit_nodes
from cubed.runtime.backup import should_launch_backup
from cubed.runtime.executors.modal import Container, run_remotely, stub
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import handle_callbacks


# We need map_unordered for the use_backups implementation
async def map_unordered(
    app_function,
    input,
    use_backups=False,
    backup_function=None,
    return_stats=False,
    name=None,
    **kwargs,
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
        async for result in app_function.map(input, order_outputs=False, kwargs=kwargs):
            if return_stats:
                result, stats = result
                if name is not None:
                    stats["array_name"] = name
                stats["task_create_tstamp"] = task_create_tstamp
                yield result, stats
            else:
                yield result
        return

    backup_function = backup_function or app_function

    tasks = {
        asyncio.ensure_future(app_function.call.aio(i, **kwargs)): i for i in input
    }
    pending = set(tasks.keys())
    t = time.monotonic()
    start_times = {f: t for f in pending}
    end_times = {}
    backups = {}

    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED, timeout=2
        )
        for task in finished:
            # TODO: use exception groups in Python 3.11 to handle case of multiple task exceptions
            if task.exception():
                # if the task has a backup that is not done, or is done with no exception, then don't raise this exception
                backup = backups.get(task, None)
                if backup:
                    if not backup.done() or not backup.exception():
                        continue
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
                    if backup in pending:
                        pending.remove(backup)
                    del backups[task]
                    del backups[backup]
                    backup.cancel()

        if use_backups:
            now = time.monotonic()
            for task in copy.copy(pending):
                if task not in backups and should_launch_backup(
                    task, now, start_times, end_times
                ):
                    # launch backup task
                    print("Launching backup task")
                    i = tasks[task]
                    new_task = asyncio.ensure_future(
                        backup_function.call.aio(i, **kwargs)
                    )
                    tasks[new_task] = i
                    start_times[new_task] = time.monotonic()
                    pending.add(new_task)
                    backups[task] = new_task
                    backups[new_task] = task


# This just retries the initial connection attempt, not the function calls
@retry(
    reraise=True,
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
)
async def async_execute_dag(
    dag, callbacks=None, array_names=None, resume=None, cloud=None, **kwargs
):
    async with stub.run():
        cloud = cloud or "aws"
        if cloud == "aws":
            app_function = run_remotely
        elif cloud == "gcp":
            app_function = Container().run_remotely
        else:
            raise ValueError(f"Unrecognized cloud: {cloud}")
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    async for _, stats in map_unordered(
                        app_function,
                        stage.mappable,
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

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute_dag(self, dag, callbacks=None, array_names=None, resume=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio.run(
            async_execute_dag(
                dag,
                callbacks=callbacks,
                array_names=array_names,
                resume=resume,
                **merged_kwargs,
            )
        )
