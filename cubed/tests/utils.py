import platform
import sys
from typing import Iterable

import networkx as nx
import numpy as np
import zarr

from cubed.runtime.create import create_executor
from cubed.runtime.types import Callback

LITHOPS_LOCAL_CONFIG = {"lithops": {"backend": "localhost", "storage": "localhost"}}

ALL_EXECUTORS = [create_executor("single-threaded")]

# don't run all tests on every executor as it's too slow, so just have a subset
MAIN_EXECUTORS = [create_executor("single-threaded")]


if platform.system() != "Windows":
    # AsyncPythonDagExecutor calls `peak_measured_mem` which is not supported on Windows
    ALL_EXECUTORS.append(create_executor("threads"))

    # AsyncPythonDagExecutor (processes) uses an API available from 3.11 onwards (max_tasks_per_child)
    if sys.version_info >= (3, 11):
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
    executor_options = dict(config=LITHOPS_LOCAL_CONFIG)
    ALL_EXECUTORS.append(create_executor("lithops", executor_options))
    MAIN_EXECUTORS.append(create_executor("lithops", executor_options))
except ImportError:
    pass

MODAL_EXECUTORS = []

try:
    MODAL_EXECUTORS.append(create_executor("modal"))
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
    za = zarr.open(
        store, mode="w", shape=a.shape, dtype=dtype, chunks=chunks, path=path
    )
    za[:] = a
    return za


def execute_pipeline(pipeline, executor):
    """Executes a pipeline"""
    dag = nx.MultiDiGraph()
    dag.add_node("node", pipeline=pipeline)
    executor.execute_dag(dag)
