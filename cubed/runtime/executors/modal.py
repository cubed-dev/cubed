import asyncio
import os
import time
from asyncio.exceptions import TimeoutError
from typing import Any, AsyncIterator, Iterable, Optional, Sequence

import modal
from aiostream import stream
from modal.exception import ConnectionError
from modal.functions import Function
from networkx import MultiDiGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.runtime.backup import use_backups_default
from cubed.runtime.executors.asyncio import async_map_unordered
from cubed.runtime.pipeline import visit_node_generations, visit_nodes
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import (
    asyncio_run,
    execute_with_stats,
    handle_callbacks,
    handle_operation_start_callbacks,
)
from cubed.spec import Spec

RUNTIME_MEMORY_MIB = 2000

app = modal.App("cubed-app")

requirements_file = os.getenv("CUBED_MODAL_REQUIREMENTS_FILE")

if requirements_file:
    image = modal.Image.debian_slim().pip_install_from_requirements(requirements_file)
    aws_image = image
    gcp_image = image
else:
    aws_image = modal.Image.debian_slim().pip_install(
        [
            "array-api-compat",
            "donfig",
            "fsspec",
            "mypy_extensions",  # for rechunker
            "ndindex",
            "networkx",
            "psutil",
            "pytest-mock",  # TODO: only needed for tests
            "s3fs",
            "tenacity",
            "toolz",
            "zarr",
        ]
    )
    gcp_image = modal.Image.debian_slim().pip_install(
        [
            "array-api-compat",
            "donfig",
            "fsspec",
            "mypy_extensions",  # for rechunker
            "ndindex",
            "networkx",
            "psutil",
            "pytest-mock",  # TODO: only needed for tests
            "gcsfs",
            "tenacity",
            "toolz",
            "zarr",
        ]
    )


def check_runtime_memory(spec):
    allowed_mem = spec.allowed_mem if spec is not None else None
    runtime_memory = RUNTIME_MEMORY_MIB * 1024 * 1024
    if allowed_mem is not None:
        if runtime_memory < allowed_mem:
            raise ValueError(
                f"Runtime memory ({runtime_memory}) is less than allowed_mem ({allowed_mem})"
            )


@app.function(
    image=aws_image,
    secrets=[modal.Secret.from_name("my-aws-secret")],
    memory=RUNTIME_MEMORY_MIB,
    retries=2,
    cloud="aws",
)
def run_remotely(input, func=None, config=None, name=None, compute_id=None):
    print(f"running remotely on {input} in {os.getenv('MODAL_REGION')}")
    # note we can't use the execution_stat decorator since it doesn't work with modal decorators
    result, stats = execute_with_stats(func, input, config=config)
    return result, stats


# For GCP we need to use a class so we can set up credentials by hooking into the container lifecycle
@app.cls(
    image=gcp_image,
    secrets=[modal.Secret.from_name("my-googlecloud-secret")],
    memory=RUNTIME_MEMORY_MIB,
    retries=2,
    cloud="gcp",
)
class Container:
    @modal.enter()
    def set_up_credentials(self):
        json = os.environ["SERVICE_ACCOUNT_JSON"]
        path = os.path.abspath("application_credentials.json")
        with open(path, "w") as f:
            f.write(json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

    @modal.method()
    def run_remotely(self, input, func=None, config=None, name=None, compute_id=None):
        print(f"running remotely on {input} in {os.getenv('MODAL_REGION')}")
        # note we can't use the execution_stat decorator since it doesn't work with modal decorators
        result, stats = execute_with_stats(func, input, config=config)
        return result, stats


# We need map_unordered for the use_backups implementation
async def map_unordered(
    app_function: Function,
    input: Iterable[Any],
    use_backups: bool = False,
    backup_function: Optional[Function] = None,
    batch_size: Optional[int] = None,
    return_stats: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    """
    Apply a function to items of an input list, yielding results as they are completed
    (which may be different to the input order).

    :param app_function: The Modal function to map over the data.
    :param input: An iterable of input data.
    :param use_backups: Whether to launch backup tasks to mitigate against slow-running tasks.
    :param kwargs: Keyword arguments to pass to the function.

    :return: Function values (and optionally stats) as they are completed, not necessarily in the input order.
    """

    if not use_backups and batch_size is None:
        task_create_tstamp = time.time()
        async for result in app_function.map(input, order_outputs=False, kwargs=kwargs):
            if return_stats:
                result, stats = result
                if name is not None:
                    stats["name"] = name
                stats["task_create_tstamp"] = task_create_tstamp
                yield result, stats
            else:
                yield result
        return

    def create_futures_func(input, **kwargs):
        return [
            (i, asyncio.ensure_future(app_function.remote.aio(i, **kwargs)))
            for i in input
        ]

    backup_function = backup_function or app_function

    def create_backup_futures_func(input, **kwargs):
        return [
            (i, asyncio.ensure_future(backup_function.remote.aio(i, **kwargs)))
            for i in input
        ]

    async for result in async_map_unordered(
        create_futures_func,
        input,
        use_backups=use_backups,
        create_backup_futures_func=create_backup_futures_func,
        batch_size=batch_size,
        return_stats=return_stats,
        name=name,
        **kwargs,
    ):
        yield result


def pipeline_to_stream(app_function, name, pipeline, **kwargs):
    return stream.iterate(
        map_unordered(
            app_function,
            pipeline.mappable,
            return_stats=True,
            name=name,
            func=pipeline.function,
            config=pipeline.config,
            **kwargs,
        )
    )


# This just retries the initial connection attempt, not the function calls
@retry(
    reraise=True,
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
)
async def async_execute_dag(
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    resume: Optional[bool] = None,
    spec: Optional[Spec] = None,
    cloud: Optional[str] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    **kwargs,
) -> None:
    if spec is not None:
        check_runtime_memory(spec)
        if "use_backups" not in kwargs and use_backups_default(spec):
            kwargs["use_backups"] = True
    async with app.run():
        cloud = cloud or "aws"
        if cloud == "aws":
            app_function = run_remotely
        elif cloud == "gcp":
            app_function = Container().run_remotely
        else:
            raise ValueError(f"Unrecognized cloud: {cloud}")
        if not compute_arrays_in_parallel:
            # run one pipeline at a time
            for name, node in visit_nodes(dag, resume=resume):
                handle_operation_start_callbacks(callbacks, name)
                st = pipeline_to_stream(app_function, name, node["pipeline"], **kwargs)
                async with st.stream() as streamer:
                    async for result, stats in streamer:
                        handle_callbacks(callbacks, result, stats)
        else:
            for gen in visit_node_generations(dag, resume=resume):
                # run pipelines in the same topological generation in parallel by merging their streams
                streams = [
                    pipeline_to_stream(app_function, name, node["pipeline"], **kwargs)
                    for name, node in gen
                ]
                merged_stream = stream.merge(*streams)
                async with merged_stream.stream() as streamer:
                    async for result, stats in streamer:
                        handle_callbacks(callbacks, result, stats)


class ModalExecutor(DagExecutor):
    """An execution engine that uses Modal's async API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "modal"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio_run(
            async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )
