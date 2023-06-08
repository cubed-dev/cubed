import os
import time
from asyncio.exceptions import TimeoutError

import modal
from modal.exception import ConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execute_with_stats, handle_callbacks

stub = modal.Stub("sync-stub")

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


# Use a generator, since we want results to be returned as they finish and we don't care about order
@stub.function(
    image=image,
    secret=modal.Secret.from_name("my-aws-secret"),
    memory=2000,
    retries=2,
    is_generator=True,
)
def run_remotely(input, func=None, config=None):
    print(f"running remotely on {input}")
    # note we can't use the execution_stat decorator since it doesn't work with modal decorators
    result, stats = execute_with_stats(func, input, config=config)
    yield result, stats


# This just retries the initial connection attempt, not the function calls
@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    stop=stop_after_attempt(3),
)
def execute_dag(dag, callbacks=None, array_names=None, resume=None, **kwargs):
    with stub.run():
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    task_create_tstamp = time.time()
                    for _, stats in run_remotely.map(
                        list(stage.mappable),
                        kwargs=dict(func=stage.function, config=pipeline.config),
                    ):
                        stats["array_name"] = name
                        stats["task_create_tstamp"] = task_create_tstamp
                        handle_callbacks(callbacks, stats)
                else:
                    raise NotImplementedError()


class ModalDagExecutor(DagExecutor):
    """An execution engine that uses Modal."""

    def execute_dag(self, dag, callbacks=None, array_names=None, **kwargs):
        execute_dag(dag, callbacks=callbacks, array_names=array_names)
