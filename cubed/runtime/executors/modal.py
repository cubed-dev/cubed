from asyncio.exceptions import TimeoutError

import modal
import networkx as nx
from modal.exception import RemoteError
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt

from cubed.runtime.types import DagExecutor

app = modal.App()

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
    return func(input, config=config)


def run_map_with_retries(input, func=None, config=None, max_attempts=3):
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
                    del tagged_inputs[tag]
    except RetryError:
        pass


def tagged_wrapper(func):
    def w(tagged_input, *args, **kwargs):
        tag, val = tagged_input
        func(val, *args, **kwargs)
        return tag

    return w


class ModalDagExecutor(DagExecutor):

    # TODO: execute tasks for independent pipelines in parallel
    @staticmethod
    def execute_dag(dag, **kwargs):
        max_attempts = 3
        try:
            for attempt in Retrying(
                retry=retry_if_exception_type(TimeoutError),
                stop=stop_after_attempt(max_attempts),
            ):
                with attempt:
                    with app.run():

                        for node in reversed(list(nx.topological_sort(dag))):
                            pipeline = nx.get_node_attributes(dag, "pipeline").get(
                                node, None
                            )
                            if pipeline is None:
                                continue

                            for stage in pipeline.stages:
                                if stage.mappable is not None:
                                    # print(f"about to run remotely on {stage.mappable}")
                                    run_map_with_retries(
                                        stage.mappable,
                                        stage.function,
                                        config=pipeline.config,
                                    )
                                else:
                                    raise NotImplementedError()
        except RetryError:
            pass
