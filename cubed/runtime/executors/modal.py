import asyncio
import os
from asyncio.exceptions import TimeoutError
from typing import Any, Callable, Optional, Sequence

import modal
from modal.exception import ConnectionError
from networkx import MultiDiGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.runtime.asyncio import async_map_dag
from cubed.runtime.backup import use_backups_default
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import asyncio_run, execute_with_stats
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


def modal_create_futures_func(function: Callable[..., Any]):
    def create_futures_func(input, **kwargs):
        return [
            (i, asyncio.ensure_future(function.remote.aio(i, **kwargs))) for i in input
        ]

    return create_futures_func


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
            self._async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )

    # This just retries the initial connection attempt, not the function calls
    @retry(
        reraise=True,
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
    )
    async def _async_execute_dag(
        self,
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

        cloud = cloud or "aws"
        if cloud == "aws":
            function = run_remotely
        elif cloud == "gcp":
            function = Container().run_remotely
        else:
            raise ValueError(f"Unrecognized cloud: {cloud}")

        async with app.run():
            create_futures_func = modal_create_futures_func(function)
            await async_map_dag(
                create_futures_func,
                dag=dag,
                callbacks=callbacks,
                resume=resume,
                compute_arrays_in_parallel=compute_arrays_in_parallel,
                **kwargs,
            )
