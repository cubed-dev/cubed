import platform

import fsspec
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.rich import RichProgressBar
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar
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


def test_resume(spec, executor):
    if executor.name == "beam":
        pytest.skip(f"{executor.name} executor does not support resume")

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)

    num_created_arrays = 2  # c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 8

    task_counter = TaskCounter()
    c.compute(executor=executor, callbacks=[task_counter], optimize_graph=False)
    num_created_arrays = 1  # c
    assert task_counter.value == num_created_arrays + 4

    # since c has already been computed, when computing d only 4 tasks are run, instead of 8
    task_counter = TaskCounter()
    d.compute(
        executor=executor, callbacks=[task_counter], optimize_graph=False, resume=True
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
    pytest.importorskip("dask.distributed")
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
    pytest.importorskip("dask.distributed")
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
        match=r"Runtime memory \(2097152000\) is less than allowed_mem \(4000000000\)",
    ):
        c.compute(executor=modal_executor)
