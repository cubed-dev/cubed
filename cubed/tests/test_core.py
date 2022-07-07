import pickle

import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed import Callback
from cubed.core.array import TqdmProgressBar
from cubed.primitive.blockwise import apply_blockwise
from cubed.runtime.executors.lithops import LithopsDagExecutor
from cubed.runtime.executors.python import PythonDagExecutor
from cubed.tests.utils import ALL_EXECUTORS, create_zarr


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


@pytest.fixture(scope="module", params=ALL_EXECUTORS)
def executor(request):
    return request.param


def test_regular_chunks(spec):
    xp.ones((5, 5), chunks=((2, 2, 1), (5,)), spec=spec)
    with pytest.raises(ValueError):
        xp.ones((5, 5), chunks=((2, 1, 2), (5,)), spec=spec)


def test_from_zarr(tmp_path, spec, executor):
    store = store = tmp_path / "source.zarr"
    create_zarr(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        store=store,
    )
    a = cubed.from_zarr(store, spec=spec)
    assert_array_equal(
        a.compute(executor=executor), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )


def test_to_zarr(tmp_path, spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    output = tmp_path / "output.zarr"
    cubed.to_zarr(a, output, executor=executor)
    res = zarr.open(output)
    assert_array_equal(res[:], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_map_blocks_with_kwargs(spec, executor):
    # based on dask test
    a = xp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], chunks=5, spec=spec)
    b = cubed.map_blocks(np.max, a, axis=0, keepdims=True, dtype=a.dtype, chunks=(1,))
    assert_array_equal(b.compute(executor=executor), np.array([4, 9]))


def test_map_blocks_with_block_id(spec, executor):
    # based on dask test
    def func(block, block_id=None, c=0):
        return np.ones_like(block) * sum(block_id) + c

    a = xp.arange(10, dtype="int64", chunks=(2,))
    b = cubed.map_blocks(func, a, dtype="int64")

    assert_array_equal(
        b.compute(executor=executor),
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype="int64"),
    )

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = cubed.map_blocks(func, a, dtype="int64")

    assert_array_equal(
        b.compute(executor=executor),
        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 2]], dtype="int64"),
    )

    c = cubed.map_blocks(func, a, dtype="int64", c=1)

    assert_array_equal(
        c.compute(executor=executor),
        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 2]], dtype="int64") + 1,
    )


def test_map_blocks_with_different_block_shapes(spec):
    def func(x, y):
        return x

    a = xp.asarray([[[12, 13]]], spec=spec)
    b = xp.asarray([14, 15], spec=spec)
    c = cubed.map_blocks(
        func, a, b, dtype="int64", chunks=(1, 1, 2), drop_axis=2, new_axis=2
    )
    assert_array_equal(c.compute(), np.array([[[12, 13]]]))


def test_multiple_ops(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)
    assert_array_equal(
        d.compute(executor=executor),
        np.array([[-2, -3, -4], [-5, -6, -7], [-8, -9, -10]]),
    )


def test_compute_is_idempotent(spec, executor):
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    assert_array_equal(b.compute(executor=executor), -np.ones((3, 3)))
    assert_array_equal(b.compute(executor=executor), -np.ones((3, 3)))


def test_default_spec(executor):
    # default spec works for small computations
    a = xp.ones((3, 3), chunks=(2, 2))
    b = xp.negative(a)
    assert_array_equal(
        b.compute(executor=executor),
        -np.ones((3, 3)),
    )


def test_default_spec_max_mem_exceeded():
    # default spec fails for large computations
    a = xp.ones((100000, 100000), chunks=(10000, 10000))
    with pytest.raises(ValueError):
        xp.negative(a)


def test_reduction_multiple_rounds(tmp_path, executor):
    spec = cubed.Spec(tmp_path, max_mem=110)
    a = xp.ones((100, 10), dtype=np.uint8, chunks=(1, 10), spec=spec)
    b = xp.sum(a, axis=0, dtype=np.uint8)
    assert_array_equal(b.compute(executor=executor), np.ones((100, 10)).sum(axis=0))


def test_visualize(tmp_path):
    a = xp.ones((100, 10), dtype=np.uint8, chunks=(1, 10))
    b = xp.sum(a, axis=0)

    assert not (tmp_path / "myplan.dot").exists()
    assert not (tmp_path / "myplan.png").exists()
    assert not (tmp_path / "myplan.svg").exists()

    b.visualize(filename=tmp_path / "myplan")
    assert (tmp_path / "myplan.svg").exists()

    b.visualize(filename=tmp_path / "myplan", format="png")
    assert (tmp_path / "myplan.png").exists()

    b.visualize(filename=tmp_path / "myplan", format="dot")
    assert (tmp_path / "myplan.dot").exists()


def test_array_pickle(spec, executor):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    c = xp.matmul(a, b)

    # we haven't computed c yet, so pickle and unpickle, and check it still works
    c = pickle.loads(pickle.dumps(c))

    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    y = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    expected = np.matmul(x, y)
    assert_array_equal(c.compute(executor=executor), expected)


mock_call_counter = 0


def mock_apply_blockwise(*args, **kwargs):
    # Raise an error on every 3rd call
    global mock_call_counter
    mock_call_counter += 1
    if mock_call_counter % 3 == 0:
        raise IOError("Test fault injection")
    return apply_blockwise(*args, **kwargs)


def test_retries(mocker, spec):
    # Inject faults into the primitive layer
    mocker.patch(
        "cubed.primitive.blockwise.apply_blockwise", side_effect=mock_apply_blockwise
    )

    executor = PythonDagExecutor()
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    )


class TaskCounter(Callback):
    def on_compute_start(self, arr):
        self.value = 0

    def on_task_end(self, name=None):
        self.value += 1


def test_callbacks(spec, executor):
    if not isinstance(executor, (PythonDagExecutor, LithopsDagExecutor)):
        pytest.skip(f"{type(executor)} does not support callbacks")

    task_counter = TaskCounter()
    progress = TqdmProgressBar()  # test indirectly (doesn't fail)

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor, callbacks=[task_counter, progress]),
        np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
    )

    assert task_counter.value == 4


def test_already_computed(spec):
    executor = PythonDagExecutor()

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)

    assert d.plan.num_tasks(d.name, optimize_graph=False) == 8

    task_counter = TaskCounter()
    c.compute(executor=executor, callbacks=[task_counter], optimize_graph=False)
    assert task_counter.value == 4

    # since c has already been computed, when computing d only 4 tasks are run, instead of 8
    task_counter = TaskCounter()
    d.compute(executor=executor, callbacks=[task_counter], optimize_graph=False)
    assert task_counter.value == 4


def test_fusion(spec):
    executor = PythonDagExecutor()
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = xp.negative(c)

    assert d.plan.num_tasks(d.name, optimize_graph=False) == 12
    assert d.plan.num_tasks(d.name, optimize_graph=True) == 4

    task_counter = TaskCounter()
    result = d.compute(executor=executor, callbacks=[task_counter])
    assert task_counter.value == 4

    assert_array_equal(
        result,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32),
    )


def test_no_fusion(spec):
    executor = PythonDagExecutor()
    # b can't be fused with c because d also depends on b
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.positive(b)
    d = xp.equal(b, c)

    assert d.plan.num_tasks(d.name, optimize_graph=False) == 3
    assert d.plan.num_tasks(d.name, optimize_graph=True) == 3

    task_counter = TaskCounter()
    result = d.compute(executor=executor, callbacks=[task_counter])
    assert task_counter.value == 3

    assert_array_equal(result, np.ones((2, 2)))
