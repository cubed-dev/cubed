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


@async_app.function(image=image, secret=modal.ref("my-aws-secret"))
async def async_run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    return func(input, config=config)


@app.function(image=image, secret=modal.ref("my-aws-secret"))
def run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    # we can't return None, as Modal map won't work in that case
    return func(input, config=config) or 1


async def map_with_retries(
    app_function, input, max_failures=3, callbacks=None, **kwargs
):
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
                if callbacks is not None:
                    [callback.on_task_end() for callback in callbacks]


def sync_map(app_function, input, callbacks=None, **kwargs):
    for _ in app_function.map(
        input,
        window=1,
        kwargs=kwargs,
    ):
        if callbacks is not None:
            [callback.on_task_end() for callback in callbacks]


def sync_execute_dag(dag, callbacks=None, **kwargs):
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
                                sync_map(
                                    run_remotely,
                                    list(stage.mappable),
                                    callbacks=callbacks,
                                    func=stage.function,
                                    config=pipeline.config,
                                )
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
                                await map_with_retries(
                                    async_run_remotely,
                                    list(stage.mappable),
                                    callbacks=callbacks,
                                    func=stage.function,
                                    config=pipeline.config,
                                )
                            else:
                                raise NotImplementedError()
    except RetryError:
        pass


class ModalDagExecutor(DagExecutor):
    # TODO: execute tasks for independent pipelines in parallel
    def execute_dag(self, dag, callbacks=None, **kwargs):
        sync_execute_dag(dag, callbacks=callbacks)


class AsyncModalDagExecutor(DagExecutor):
    def execute_dag(self, dag, callbacks=None, **kwargs):
        asyncio.run(async_execute_dag(dag, callbacks=callbacks))
