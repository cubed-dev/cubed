import os
import time
from asyncio.exceptions import TimeoutError

import modal
from modal.exception import ConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execute_with_stats, handle_callbacks

stub = modal.Stub("cubed-stub")

requirements_file = os.getenv("CUBED_MODAL_REQUIREMENTS_FILE")

if requirements_file:
    image = modal.Image.debian_slim().pip_install_from_requirements(requirements_file)
    aws_image = image
    gcp_image = image
else:
    aws_image = modal.Image.debian_slim().pip_install(
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
    gcp_image = modal.Image.debian_slim().pip_install(
        [
            "fsspec",
            "mypy_extensions",  # for rechunker
            "networkx",
            "pytest-mock",  # TODO: only needed for tests
            "gcsfs",
            "tenacity",
            "toolz",
            "zarr",
        ]
    )


@stub.function(
    image=aws_image,
    secret=modal.Secret.from_name("my-aws-secret"),
    memory=2000,
    retries=2,
    cloud="aws",
)
def run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    # note we can't use the execution_stat decorator since it doesn't work with modal decorators
    result, stats = execute_with_stats(func, input, config=config)
    return result, stats


# For GCP we need to use a class so we can set up credentials by hooking into the container lifecycle
@stub.cls(
    image=gcp_image,
    secret=modal.Secret.from_name("my-googlecloud-secret"),
    memory=2000,
    retries=2,
    cloud="gcp",
)
class Container:
    def __enter__(self):
        json = os.environ["SERVICE_ACCOUNT_JSON"]
        path = os.path.abspath("application_credentials.json")
        with open(path, "w") as f:
            f.write(json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

    @modal.method()
    def run_remotely(self, input, func=None, config=None):
        print(f"running remotely on {input}")
        # note we can't use the execution_stat decorator since it doesn't work with modal decorators
        result, stats = execute_with_stats(func, input, config=config)
        return result, stats


# This just retries the initial connection attempt, not the function calls
@retry(
    reraise=True,
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
)
def execute_dag(
    dag, callbacks=None, array_names=None, resume=None, cloud=None, **kwargs
):
    with stub.run():
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
                    task_create_tstamp = time.time()
                    for _, stats in app_function.map(
                        stage.mappable,
                        order_outputs=False,
                        kwargs=dict(func=stage.function, config=pipeline.config),
                    ):
                        stats["array_name"] = name
                        stats["task_create_tstamp"] = task_create_tstamp
                        handle_callbacks(callbacks, stats)
                else:
                    raise NotImplementedError()


class ModalDagExecutor(DagExecutor):
    """An execution engine that uses Modal."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute_dag(self, dag, callbacks=None, array_names=None, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}
        execute_dag(dag, callbacks=callbacks, array_names=array_names, **merged_kwargs)
