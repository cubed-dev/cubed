import time
from asyncio.exceptions import TimeoutError

import modal
from modal.exception import ConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.core.array import TaskEndEvent
from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor
from cubed.utils import peak_memory

stub = modal.Stub()

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


# Use a generator, since we want results to be returned as they finish and we don't care about order
@stub.generator(image=image, secret=modal.ref("my-aws-secret"), retries=2)
def run_remotely(input, func=None, config=None):
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


# This just retries the initial connection attempt, not the function calls
@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
)
def execute_dag(dag, callbacks=None, **kwargs):
    with stub.run():
        for name, node in visit_nodes(dag):
            pipeline = node["pipeline"]

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    task_create_tstamp = time.time()
                    for result in run_remotely.map(
                        list(stage.mappable),
                        kwargs=dict(func=stage.function, config=pipeline.config),
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


class ModalDagExecutor(DagExecutor):
    """An execution engine that uses Modal."""

    def execute_dag(self, dag, callbacks=None, **kwargs):
        execute_dag(dag, callbacks=callbacks)
