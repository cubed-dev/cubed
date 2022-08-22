from typing import Iterable

import numpy as np
import zarr
from rechunker.executors.python import PythonPipelineExecutor

from cubed.runtime.executors.python import PythonDagExecutor
from cubed.runtime.executors.python_async import AsyncPythonDagExecutor

LITHOPS_LOCAL_CONFIG = {"lithops": {"backend": "localhost", "storage": "localhost"}}

ALL_EXECUTORS = [
    PythonPipelineExecutor(),
    PythonDagExecutor(),
    AsyncPythonDagExecutor(),
]

# don't run all tests on every executor as it's too slow, so just have a subset
MAIN_EXECUTORS = [PythonPipelineExecutor(), PythonDagExecutor()]

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
