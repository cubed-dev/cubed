import asyncio
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


async def map_as_completed(app_function, input, max_failures=3, **kwargs):
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    :param app_function: The Modal function to map over the data.
    :param input: An iterable of input data.
    :param max_failures: The number of task failures to allow before raising an exception.
    :param kwargs: Keyword arguments to pass to the function.

    :return: Function values as they are completed, not necessarily in the input order.
    """
    failures = 0
    tasks = {asyncio.ensure_future(app_function(i, **kwargs)): i for i in input}
    pending = set(tasks.keys())
    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )

        for task in finished:
            if task.exception():
                failures += 1
                if failures > max_failures:
                    raise task.exception()
                i = tasks[task]
                new_task = asyncio.ensure_future(app_function(i, **kwargs))
                tasks[new_task] = i
                pending.add(new_task)
            else:
                yield task.result()


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
