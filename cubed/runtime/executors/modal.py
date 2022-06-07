import asyncio
from asyncio.exceptions import TimeoutError

import modal
import modal.aio
import networkx as nx
from modal.exception import RemoteError
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt

from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import DagExecutor

app = modal.aio.AioApp()

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
async def run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    return func(input, config=config)


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


def run_map_with_retries(input, func=None, config=None, max_attempts=3, callbacks=None):
    tagged_inputs = {k: v for (k, v) in enumerate(input)}

    try:
        for attempt in Retrying(
            retry=retry_if_exception_type((RemoteError, OSError)),
            stop=stop_after_attempt(max_attempts),
        ):
            with attempt:
                if len(tagged_inputs) == 0:
                    break
                for tag in run_remotely.map(
                    list(tagged_inputs.items()),
                    window=1,
                    kwargs=dict(func=tagged_wrapper(func), config=config),
                ):
                    print(f"Completed tag {tag}")
                    if callbacks is not None:
                        [callback.on_task_end() for callback in callbacks]
                    del tagged_inputs[tag]
    except RetryError:
        pass


def tagged_wrapper(func):
    def w(tagged_input, *args, **kwargs):
        tag, val = tagged_input
        func(val, *args, **kwargs)
        return tag

    return w


async def execute_dag_async(dag, callbacks=None, **kwargs):
    async with app.run():
        nodes = {n: d for (n, d) in dag.nodes(data=True)}
        for node in reversed(list(nx.topological_sort(dag))):
            if already_computed(nodes[node]):
                continue
            pipeline = nodes[node]["pipeline"]

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    # print(f"about to run remotely on {stage.mappable}")
                    await map_with_retries(
                        run_remotely,
                        stage.mappable,
                        callbacks=callbacks,
                        func=stage.function,
                        config=pipeline.config,
                    )
                else:
                    raise NotImplementedError()


class ModalDagExecutor(DagExecutor):

    # TODO: execute tasks for independent pipelines in parallel
    @staticmethod
    def execute_dag(dag, callbacks=None, **kwargs):
        asyncio.run(execute_dag_async(dag, callbacks=callbacks))

        # max_attempts = 3
        # try:
        #     for attempt in Retrying(
        #         retry=retry_if_exception_type(TimeoutError),
        #         stop=stop_after_attempt(max_attempts),
        #     ):
        #         with attempt:
        #             with app.run():

        #                 nodes = {n: d for (n, d) in dag.nodes(data=True)}
        #                 for node in reversed(list(nx.topological_sort(dag))):
        #                     if already_computed(nodes[node]):
        #                         continue
        #                     pipeline = nodes[node]["pipeline"]

        #                     for stage in pipeline.stages:
        #                         if stage.mappable is not None:
        #                             # print(f"about to run remotely on {stage.mappable}")
        #                             run_map_with_retries(
        #                                 stage.mappable,
        #                                 stage.function,
        #                                 config=pipeline.config,
        #                                 callbacks=callbacks,
        #                             )
        #                         else:
        #                             raise NotImplementedError()
        # except RetryError:
        #     pass
