import platform
from typing import Iterable

import networkx as nx
import numpy as np

from cubed import config
from cubed.runtime.create import create_executor
from cubed.runtime.types import Callback
from cubed.storage.backend import open_backend_array

LITHOPS_LOCAL_CONFIG = {
    "lithops": {
        "backend": "localhost",
        "storage": "localhost",
        "monitoring_interval": 0.1,
        "include_modules": None,
    },
    "localhost": {"version": 1},
}

ALL_EXECUTORS = [create_executor("single-threaded")]

# don't run all tests on every executor as it's too slow, so just have a subset
MAIN_EXECUTORS = [create_executor("single-threaded")]


if platform.system() != "Windows":
    # ThreadsExecutor calls `peak_measured_mem` which is not supported on Windows
    ALL_EXECUTORS.append(create_executor("threads"))
    MAIN_EXECUTORS.append(create_executor("threads"))

    ALL_EXECUTORS.append(create_executor("processes"))
    MAIN_EXECUTORS.append(create_executor("processes"))

try:
    ALL_EXECUTORS.append(create_executor("beam"))
    MAIN_EXECUTORS.append(create_executor("beam"))
except ImportError:
    pass

try:
    ALL_EXECUTORS.append(create_executor("dask"))
    MAIN_EXECUTORS.append(create_executor("dask"))
except ImportError:
    pass

try:
    executor_options = dict(config=LITHOPS_LOCAL_CONFIG, wait_dur_sec=0.1)
    ALL_EXECUTORS.append(create_executor("lithops", executor_options))
    MAIN_EXECUTORS.append(create_executor("lithops", executor_options))
except ImportError:
    pass

try:
    ALL_EXECUTORS.append(create_executor("ray"))
    MAIN_EXECUTORS.append(create_executor("ray"))
except ImportError:
    pass


MODAL_EXECUTORS = []

try:
    # only set global config below if modal can be imported
    import modal  # noqa: F401

    # need to set global config for testing modal since these options
    # are read at the top level of modal.py to create remote functions
    config.set(
        {
            "spec.executor_options.cloud": "aws",
            "spec.executor_options.region": "us-east-1",
        }
    )
    executor_options = dict(enable_output=True)
    MODAL_EXECUTORS.append(create_executor("modal", executor_options))
except ImportError:
    pass


class TaskCounter(Callback):
    def __init__(self, check_timestamps=True) -> None:
        self.check_timestamps = check_timestamps

    def on_compute_start(self, event):
        self.value = 0

    def on_task_end(self, event):
        if self.check_timestamps and event.task_create_tstamp is not None:
            assert (
                event.task_result_tstamp
                >= event.function_end_tstamp
                >= event.function_start_tstamp
                >= event.task_create_tstamp
                > 0
            )
        self.value += event.num_tasks


def create_zarr(a, /, store, *, dtype=None, chunks=None, path=None):
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    # write to zarr
    za = open_backend_array(
        store, mode="w", shape=a.shape, dtype=dtype, chunks=chunks, path=path
    )
    za[:] = a
    return za


def execute_pipeline(pipeline, executor):
    """Executes a pipeline"""
    dag = nx.MultiDiGraph()
    dag.add_node("node", pipeline=pipeline)
    executor.execute_dag(dag)
