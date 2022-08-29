import pickle

import fsspec
import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed import Callback
from cubed.core.plan import num_tasks
from cubed.extensions.tqdm import TqdmProgressBar
from cubed.primitive.blockwise import apply_blockwise
from cubed.runtime.executors.python import PythonDagExecutor
from cubed.tests.utils import MAIN_EXECUTORS, MODAL_EXECUTORS, create_zarr


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


@pytest.fixture(scope="module", params=MAIN_EXECUTORS)
def executor(request):
    return request.param


@pytest.fixture(scope="module", params=MODAL_EXECUTORS)
def modal_executor(request):
    return request.param


def test_regular_chunks(spec):
    xp.ones((5, 5), chunks=((2, 2, 1), (5,)), spec=spec)
    with pytest.raises(ValueError):
        xp.ones((5, 5), chunks=((2, 1, 2), (5,)), spec=spec)


class WrappedArray:
    def __init__(self, x):
        self.x = x
        self.dtype = x.dtype
        self.shape = x.shape
        self.ndim = len(x.shape)

    def __array__(self, dtype=None):
        return np.asarray(self.x, dtype=dtype)

    def __getitem__(self, i):
        return WrappedArray(self.x[i])


@pytest.mark.parametrize(
    "x,chunks,asarray",
    [
        (np.arange(25).reshape((5, 5)), (5, 5), None),
        (np.arange(25).reshape((5, 5)), (3, 2), True),
        (np.arange(25).reshape((5, 5)), -1, True),
        (np.array([[1]]), 1, None),
    ],
)
def test_from_array(x, chunks, asarray):
    a = cubed.from_array(WrappedArray(x), chunks=chunks, asarray=asarray)
    assert_array_equal(a, x)


def test_from_array_zarr(tmp_path, spec):
    store = store = tmp_path / "source.zarr"
    za = create_zarr(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        store=store,
    )
    a = cubed.from_array(za, spec=spec)
    assert_array_equal(a, za)


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


def test_store(tmp_path, spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)

    store = tmp_path / "source.zarr"
    target = zarr.empty(a.shape, store=store)

    cubed.store(a, target)
    assert_array_equal(target, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_store_multiple(tmp_path, spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)

    store1 = tmp_path / "source1.zarr"
    target1 = zarr.empty(a.shape, store=store1)
    store2 = tmp_path / "source2.zarr"
    target2 = zarr.empty(b.shape, store=store2)

    cubed.store([a, b], [target1, target2])
    assert_array_equal(target1, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert_array_equal(target2, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))


def test_store_fails(tmp_path, spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    store = tmp_path / "source.zarr"
    target = zarr.empty(a.shape, store=store)

    with pytest.raises(
        ValueError, match=r"Different number of sources \(2\) and targets \(1\)"
    ):
        cubed.store([a, b], [target])

    with pytest.raises(ValueError, match="All sources must be cubed array objects"):
        cubed.store([1], [target])


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


def test_rechunk(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 1), spec=spec)
    b = a.rechunk((1, 2))
    assert_array_equal(
        b.compute(executor=executor),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
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


def test_different_specs(tmp_path):
    spec1 = cubed.Spec(tmp_path, max_mem=100000)
    spec2 = cubed.Spec(tmp_path, max_mem=200000)
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec1)
    b = xp.ones((3, 3), chunks=(2, 2), spec=spec2)
    with pytest.raises(ValueError):
        xp.add(a, b)


def test_reduction_multiple_rounds(tmp_path, executor):
    spec = cubed.Spec(tmp_path, max_mem=1000)
    a = xp.ones((100, 10), dtype=np.uint8, chunks=(1, 10), spec=spec)
    b = xp.sum(a, axis=0, dtype=np.uint8)
    # check that there is > 1 rechunk step
    rechunks = [
        n for (n, d) in b.plan.dag.nodes(data=True) if d["op_name"] == "rechunk"
    ]
    assert len(rechunks) > 1
    assert_array_equal(b.compute(executor=executor), np.ones((100, 10)).sum(axis=0))


def test_reduction_not_enough_memory(tmp_path):
    spec = cubed.Spec(tmp_path, max_mem=50)
    a = xp.ones((100, 10), dtype=np.uint8, chunks=(1, 10), spec=spec)
    with pytest.raises(ValueError, match=r"Not enough memory for reduction"):
        xp.sum(a, axis=0, dtype=np.uint8)


def test_compute_multiple():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2))
    c = xp.add(a, b)
    d = c * 2
    e = c * 3

    f = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    g = f * 4

    dc, ec, gc = cubed.compute(d, e, g)

    an = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bn = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    cn = an + bn
    dn = cn * 2
    en = cn * 3

    fn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    gn = fn * 4

    assert_array_equal(dc, dn)
    assert_array_equal(ec, en)
    assert_array_equal(gc, gn)


def test_compute_multiple_different_specs(tmp_path):
    spec1 = cubed.Spec(tmp_path, max_mem=100000)
    spec2 = cubed.Spec(tmp_path, max_mem=200000)

    a1 = xp.ones((3, 3), chunks=(2, 2), spec=spec1)
    b1 = xp.ones((3, 3), chunks=(2, 2), spec=spec1)
    c1 = xp.add(a1, b1)

    a2 = xp.ones((3, 3), chunks=(2, 2), spec=spec2)
    b2 = xp.ones((3, 3), chunks=(2, 2), spec=spec2)
    c2 = xp.add(a2, b2)

    with pytest.raises(ValueError):
        cubed.compute(c1, c2)


def test_visualize(tmp_path):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2))
    c = xp.add(a, b)
    d = c * 2
    e = c * 3

    f = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    g = f * 4

    assert not (tmp_path / "e.dot").exists()
    assert not (tmp_path / "e.png").exists()
    assert not (tmp_path / "e.svg").exists()
    assert not (tmp_path / "dg.svg").exists()

    e.visualize(filename=tmp_path / "e")
    assert (tmp_path / "e.svg").exists()

    e.visualize(filename=tmp_path / "e", format="png")
    assert (tmp_path / "e.png").exists()

    e.visualize(filename=tmp_path / "e", format="dot")
    assert (tmp_path / "e.dot").exists()

    # multiple arrays
    cubed.visualize(d, g, filename=tmp_path / "dg")
    assert (tmp_path / "dg.svg").exists()


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
    def on_compute_start(self, dag):
        self.value = 0

    def on_task_end(self, event):
        if event.task_create_tstamp is not None:
            assert (
                event.task_result_tstamp
                >= event.function_end_tstamp
                >= event.function_start_tstamp
                >= event.task_create_tstamp
                > 0
            )
        self.value += 1


def test_callbacks(spec, executor):
    from cubed.runtime.executors.lithops import LithopsDagExecutor

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


@pytest.mark.cloud
def test_callbacks_modal(spec, modal_executor):
    task_counter = TaskCounter()
    tmp_path = "s3://cubed-unittest/callbacks"
    spec = cubed.Spec(tmp_path, max_mem=100000)
    try:
        a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
        b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
        c = xp.add(a, b)
        assert_array_equal(
            c.compute(executor=modal_executor, callbacks=[task_counter]),
            np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
        )

        assert task_counter.value == 4
    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


def test_already_computed(spec):
    executor = PythonDagExecutor()

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)

    assert num_tasks(d.plan.dag, optimize_graph=False) == 8

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

    assert num_tasks(d.plan.dag, optimize_graph=False) == 12
    assert num_tasks(d.plan.dag, optimize_graph=True) == 4

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

    assert num_tasks(d.plan.dag, optimize_graph=False) == 3
    assert num_tasks(d.plan.dag, optimize_graph=True) == 3

    task_counter = TaskCounter()
    result = d.compute(executor=executor, callbacks=[task_counter])
    assert task_counter.value == 3

    assert_array_equal(result, np.ones((2, 2)))


def test_no_fusion_multiple_edges(spec):
    executor = PythonDagExecutor()
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.asarray(b)
    # b and c are the same array, so d has a single dependency
    # with multiple edges
    # this should not be fused under the current logic
    d = xp.equal(b, c)

    assert num_tasks(d.plan.dag, optimize_graph=False) == 2
    assert num_tasks(d.plan.dag, optimize_graph=True) == 2

    task_counter = TaskCounter()
    result = d.compute(executor=executor, callbacks=[task_counter])
    assert task_counter.value == 2

    assert_array_equal(result, np.full((2, 2), True))


def test_measure_baseline_memory(executor):
    from cubed.runtime.executors.lithops import LithopsDagExecutor

    if not isinstance(executor, LithopsDagExecutor):
        pytest.skip(f"{type(executor)} does not support measure_baseline_memory")

    baseline_memory = cubed.measure_baseline_memory(executor=executor)
    assert baseline_memory > 1_000_000  # over 1MB
