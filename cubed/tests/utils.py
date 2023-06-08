import platform
from typing import Iterable

import numpy as np
import zarr

from cubed.core.array import Callback
from cubed.runtime.executors.python import PythonDagExecutor
from cubed.runtime.executors.python_async import AsyncPythonDagExecutor
from cubed.vendor.rechunker.executors.python import PythonPipelineExecutor

LITHOPS_LOCAL_CONFIG = {"lithops": {"backend": "localhost", "storage": "localhost"}}

ALL_EXECUTORS = [
    PythonPipelineExecutor(),
    PythonDagExecutor(),
]

# don't run all tests on every executor as it's too slow, so just have a subset
MAIN_EXECUTORS = [PythonPipelineExecutor(), PythonDagExecutor()]


if platform.system() != "Windows":
    # AsyncPythonDagExecutor calls `peak_measured_mem` which is not supported on Windows
    ALL_EXECUTORS.append(AsyncPythonDagExecutor())


try:
    from cubed.runtime.executors.beam import BeamDagExecutor, BeamPipelineExecutor

    ALL_EXECUTORS.append(BeamDagExecutor())
    ALL_EXECUTORS.append(BeamPipelineExecutor())

    MAIN_EXECUTORS.append(BeamDagExecutor())
except ImportError:
    pass

try:
    from cubed.runtime.executors.lithops import (
        LithopsDagExecutor,
        LithopsPipelineExecutor,
    )

    ALL_EXECUTORS.append(LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG))
    ALL_EXECUTORS.append(LithopsPipelineExecutor(config=LITHOPS_LOCAL_CONFIG))

    MAIN_EXECUTORS.append(LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG))
except ImportError:
    pass

MODAL_EXECUTORS = []

try:
    from cubed.runtime.executors.modal import ModalDagExecutor
    from cubed.runtime.executors.modal_async import AsyncModalDagExecutor

    MODAL_EXECUTORS.append(AsyncModalDagExecutor())
    MODAL_EXECUTORS.append(ModalDagExecutor())
except ImportError:
    pass


class TaskCounter(Callback):
    def __init__(self, check_timestamps=True) -> None:
        self.check_timestamps = check_timestamps

    def on_compute_start(self, dag, resume):
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


def create_zarr(a, /, store, *, dtype=None, chunks=None):
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    # write to zarr
    za = zarr.open(store, mode="w", shape=a.shape, dtype=dtype, chunks=chunks)
    za[:] = a
    return za


def execute_pipeline(pipeline, executor):
    """Executes a pipeline"""
    plan = executor.pipelines_to_plan([pipeline])
    executor.execute_plan(plan)
