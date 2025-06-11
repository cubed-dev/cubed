import contextlib
import os
import platform
import re

import fsspec
import numpy as np
import psutil
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.diagnostics import ProgressBar
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.mem_warn import MemoryWarningCallback
from cubed.diagnostics.rich import RichProgressBar
from cubed.diagnostics.timeline import TimelineVisualizationCallback
from cubed.diagnostics.tqdm import TqdmProgressBar
from cubed.primitive.blockwise import apply_blockwise
from cubed.runtime.create import create_executor
from cubed.tests.utils import (
    ALL_EXECUTORS,
    MAIN_EXECUTORS,
    MODAL_EXECUTORS,
    TaskCounter,
)


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


@pytest.fixture(
    scope="module",
    params=MAIN_EXECUTORS,
    ids=[executor.name for executor in MAIN_EXECUTORS],
)
def executor(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=ALL_EXECUTORS,
    ids=[executor.name for executor in ALL_EXECUTORS],
)
def any_executor(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=MODAL_EXECUTORS,
    ids=[executor.name for executor in MODAL_EXECUTORS],
)
def modal_executor(request):
    return request.param


mock_call_counter = 0


def mock_apply_blockwise(*args, **kwargs):
    # Raise an error on every 3rd call
    global mock_call_counter
    mock_call_counter += 1
    if mock_call_counter % 3 == 0:
        raise IOError("Test fault injection")
    return apply_blockwise(*args, **kwargs)


# see tests/runtime for more tests for retries for other executors
@pytest.mark.skipif(
    platform.system() == "Windows", reason="measuring memory does not run on windows"
)
def test_retries(mocker, spec):
    # Use threads executor since single-threaded executor doesn't support retries
    executor = create_executor("threads")
    # Inject faults into the primitive layer
    mocker.patch(
        "cubed.primitive.blockwise.apply_blockwise", side_effect=mock_apply_blockwise
    )

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    )


@pytest.mark.skipif(
    platform.system() == "Windows", reason="measuring memory does not run on windows"
)
def test_callbacks(spec, executor):
    task_counter = TaskCounter()
    # test following indirectly by checking they don't cause a failure
    progress = TqdmProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(
            executor=executor, callbacks=[task_counter, progress, hist, timeline_viz]
        ),
        np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
    )

    num_created_arrays = 1
    assert task_counter.value == num_created_arrays + 4


def test_callbacks_as_context_managers(spec, executor):
    with TaskCounter() as task_counter, ProgressBar():
        assert task_counter is not None
        a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
        b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
        c = xp.add(a, b)
        assert_array_equal(
            c.compute(executor=executor),
            np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
        )

        num_created_arrays = 1
        assert task_counter.value == num_created_arrays + 4


def test_rich_progress_bar(spec, executor):
    # test indirectly by checking it doesn't cause a failure
    progress = RichProgressBar()

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor, callbacks=[progress]),
        np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
    )


@pytest.mark.cloud
def test_callbacks_modal(spec, modal_executor):
    task_counter = TaskCounter(check_timestamps=False)
    tmp_path = "s3://cubed-unittest/callbacks"
    spec = cubed.Spec(tmp_path, allowed_mem=100000)
    try:
        a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
        b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
        c = xp.add(a, b)
        assert_array_equal(
            c.compute(executor=modal_executor, callbacks=[task_counter]),
            np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
        )

        num_created_arrays = 1
        assert task_counter.value == num_created_arrays + 4
    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="measuring memory does not run on windows"
)
def test_mem_warn(tmp_path, executor):
    if executor.name not in ("processes", "lithops"):
        pytest.skip(f"{executor.name} executor does not support MemoryWarningCallback")

    spec = cubed.Spec(tmp_path, allowed_mem=200_000_000, reserved_mem=100_000_000)
    mem_warn = MemoryWarningCallback()

    def func(a):
        np.ones(100_000_000)  # blow memory
        return a

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = cubed.map_blocks(func, a, dtype=a.dtype)
    with pytest.raises(
        UserWarning, match="Peak memory usage exceeded allowed_mem when running tasks"
    ):
        b.compute(executor=executor, callbacks=[mem_warn])


def test_resume(spec, executor):
    if executor.name == "beam":
        pytest.skip(f"{executor.name} executor does not support resume")

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)

    num_created_arrays = 2  # c, d
    assert d.plan._finalize(optimize_graph=False).num_tasks() == num_created_arrays + 8

    task_counter = TaskCounter()
    c.compute(executor=executor, callbacks=[task_counter], optimize_graph=False)
    num_created_arrays = 1  # c
    assert task_counter.value == num_created_arrays + 4

    if not hasattr(c.zarray, "nchunks_initialized"):
        # We expect resume to fail if there is no 'nchunks_initialized' property on the Zarr array
        cm = pytest.raises(NotImplementedError)
    else:
        cm = contextlib.nullcontext()

    with cm:
        # since c has already been computed, when computing d only 4 tasks are run, instead of 8
        task_counter = TaskCounter()
        d.compute(
            executor=executor,
            callbacks=[task_counter],
            optimize_graph=False,
            resume=True,
        )
        # the create arrays tasks are run again, even though they exist
        num_created_arrays = 2  # c, d
        assert task_counter.value == num_created_arrays + 4


@pytest.mark.parametrize("compute_arrays_in_parallel", [True, False])
def test_compute_arrays_in_parallel(spec, any_executor, compute_arrays_in_parallel):
    if any_executor.name == "beam":
        pytest.skip(
            f"{any_executor.name} executor does not support compute_arrays_in_parallel"
        )

    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    c = xp.add(a, b)

    # note that this merely checks that compute_arrays_in_parallel is accepted
    c.compute(
        executor=any_executor, compute_arrays_in_parallel=compute_arrays_in_parallel
    )


@pytest.mark.cloud
@pytest.mark.parametrize("compute_arrays_in_parallel", [True, False])
def test_compute_arrays_in_parallel_modal(modal_executor, compute_arrays_in_parallel):
    tmp_path = "s3://cubed-unittest/parallel_pipelines"
    spec = cubed.Spec(tmp_path, allowed_mem=100000)
    try:
        a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
        b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
        c = xp.add(a, b)

        # note that this merely checks that compute_arrays_in_parallel is accepted
        c.compute(
            executor=modal_executor,
            compute_arrays_in_parallel=compute_arrays_in_parallel,
        )
    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


def test_check_runtime_memory_dask(spec, executor):
    if executor.name != "dask":
        pytest.skip(f"{executor.name} executor does not support check_runtime_memory")

    spec = cubed.Spec(spec.work_dir, allowed_mem="4GB")  # larger than runtime memory
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    with pytest.raises(
        ValueError,
        match=r"Runtime memory \(2000000000\) is less than allowed_mem \(4000000000\)",
    ):
        c.compute(
            executor=executor,
            compute_kwargs=dict(memory_limit="2 GB", threads_per_worker=1),
        )


def test_check_runtime_memory_dask_no_workers(spec, executor):
    if executor.name != "dask":
        pytest.skip(f"{executor.name} executor does not support check_runtime_memory")

    spec = cubed.Spec(spec.work_dir, allowed_mem=100000)
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    with pytest.raises(ValueError, match=r"Cluster has no workers running"):
        c.compute(
            executor=executor,
            compute_kwargs=dict(n_workers=0),
        )


@pytest.mark.cloud
def test_check_runtime_memory_modal(spec, modal_executor):
    tmp_path = "s3://cubed-unittest/check-runtime-memory"
    spec = cubed.Spec(tmp_path, allowed_mem="4GB")  # larger than Modal runtime memory
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    with pytest.raises(
        ValueError,
        match=r"Runtime memory \(2000000000\) is less than allowed_mem \(4000000000\)",
    ):
        c.compute(executor=modal_executor)


def test_check_runtime_memory_processes(spec, executor):
    if executor.name != "processes":
        pytest.skip(f"{executor.name} executor does not support check_runtime_memory")

    total_mem = psutil.virtual_memory().total
    max_workers = os.cpu_count()
    mem_per_worker = total_mem // max_workers
    allowed_mem = mem_per_worker * 2  # larger than will fit

    spec = cubed.Spec(spec.work_dir, allowed_mem=allowed_mem)
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Total memory on machine ({total_mem}) is less than allowed_mem * max_workers ({allowed_mem} * {max_workers} = {allowed_mem * max_workers})"
        ),
    ):
        c.compute(executor=executor)

    # OK if we use fewer workers
    c.compute(executor=executor, max_workers=max_workers // 2)


COMPILE_FUNCTIONS = [lambda fn: fn]

try:
    from numba import jit as numba_jit

    COMPILE_FUNCTIONS.append(numba_jit)
except ModuleNotFoundError:
    pass

try:
    if "jax" in os.environ.get("CUBED_BACKEND_ARRAY_API_MODULE", ""):
        from jax import jit as jax_jit

        COMPILE_FUNCTIONS.append(jax_jit)
except ModuleNotFoundError:
    pass


@pytest.mark.parametrize("compile_function", COMPILE_FUNCTIONS)
def test_check_compilation(spec, executor, compile_function):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor, compile_function=compile_function),
        np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
    )


def test_compilation_can_fail(spec, executor):
    def compile_function(func):
        raise NotImplementedError(f"Cannot compile {func}")

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    with pytest.raises(NotImplementedError) as excinfo:
        c.compute(executor=executor, compile_function=compile_function)

    assert "add" in str(excinfo.value), "Compile function was applied to add operation."


def test_compilation_with_config_can_fail(spec, executor):
    def compile_function(func, *, config=None):
        raise NotImplementedError(f"Cannot compile {func} with {config}")

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    with pytest.raises(NotImplementedError) as excinfo:
        c.compute(executor=executor, compile_function=compile_function)

    assert "BlockwiseSpec" in str(
        excinfo.value
    ), "Compile function was applied with a config argument."
