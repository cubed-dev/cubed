import asyncio
import contextlib
import os
from asyncio.exceptions import TimeoutError
from typing import Any, Callable, Optional, Sequence

import modal
from modal.exception import ConnectionError
from networkx import MultiDiGraph
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed import config
from cubed.runtime.asyncio import async_map_dag
from cubed.runtime.backup import use_backups_default
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import asyncio_run, execute_with_stats
from cubed.spec import Spec, spec_from_config
from cubed.utils import convert_to_bytes

app = modal.App("cubed-app", include_source=True)

# Get Modal App settings from Cubed configuration. Note that we have to do this
# globally, since Modal remote functions have to be defined globally.
if modal.is_local():
    spec = spec_from_config(config)
    executor_options = spec.executor_options
else:
    executor_options = {}

runtime_memory = convert_to_bytes(executor_options.get("memory", "2GB"))
runtime_memory_mib = runtime_memory // 1_000_000
retries = executor_options.get("retries", 2)
timeout = executor_options.get("timeout", 180)
cloud = executor_options.get("cloud", "aws")
region = executor_options.get("region", None)
if modal.is_local() and region is None:
    raise ValueError(
        "Must set region when running using Modal, via the Cubed 'spec.executor_options.region' configuration setting."
    )

requirements_file = executor_options.get("requirements_file", None)

if requirements_file:
    image = modal.Image.debian_slim().pip_install_from_requirements(requirements_file)
else:
    if cloud == "aws":
        image = modal.Image.debian_slim().pip_install(
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
    elif cloud == "gcp":
        image = modal.Image.debian_slim().pip_install(
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
    else:
        raise ValueError(f"Unrecognized cloud: {cloud}")

if cloud == "aws":
    secrets = [modal.Secret.from_name("my-aws-secret")]
elif cloud == "gcp":
    secrets = [modal.Secret.from_name("my-googlecloud-secret")]
else:
    raise ValueError(f"Unrecognized cloud: {cloud}")


def check_runtime_memory(spec):
    allowed_mem = spec.allowed_mem if spec is not None else None
    if allowed_mem is not None:
        if runtime_memory < allowed_mem:
            raise ValueError(
                f"Runtime memory ({runtime_memory}) is less than allowed_mem ({allowed_mem})"
            )


# We use a class so for GCP we can set up credentials by hooking into the container lifecycle
@app.cls(
    image=image,
    secrets=secrets,
    cpu=1.0,
    memory=runtime_memory_mib,  # modal memory is in MiB
    retries=retries,
    timeout=timeout,
    cloud=cloud,
    region=region,
)
class Container:
    @modal.enter()
    def set_up_credentials(self):
        if os.getenv("MODAL_CLOUD_PROVIDER") == "CLOUD_PROVIDER_GCP":
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
        # remove executor options as they should already have been used in defining the remote functions
        for executor_option in ("memory", "retries", "timeout", "cloud", "region"):
            merged_kwargs.pop(executor_option, None)
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

        function = Container().run_remotely

        enable_output = kwargs.pop("enable_output", False)
        cm = modal.enable_output() if enable_output else contextlib.nullcontext()

        with cm:
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
