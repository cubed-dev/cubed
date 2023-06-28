import platform

import dill
import fsspec
import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar
from cubed.primitive.blockwise import apply_blockwise
from cubed.runtime.types import DagExecutor
from cubed.tests.utils import MAIN_EXECUTORS, MODAL_EXECUTORS, TaskCounter, create_zarr


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


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
    assert isinstance(a, cubed.Array)
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


@pytest.mark.parametrize("new_chunks", [(1, 2), {0: 1, 1: 2}])
def test_rechunk(spec, executor, new_chunks):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 1), spec=spec)
    b = a.rechunk(new_chunks)
    assert_array_equal(
        b.compute(executor=executor),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )


def test_rechunk_same_chunks(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 1), spec=spec)
    b = a.rechunk((2, 1))
    task_counter = TaskCounter()
    res = b.compute(callbacks=[task_counter])
    # no tasks except array creation task should have run since chunks are same
    num_created_arrays = 1
    assert task_counter.value == num_created_arrays

    assert_array_equal(res, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


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


def test_default_spec_allowed_mem_exceeded():
    # default spec fails for large computations
    a = xp.ones((100000, 100000), chunks=(10000, 10000))
    with pytest.raises(ValueError):
        xp.negative(a)


def test_different_specs(tmp_path):
    spec1 = cubed.Spec(tmp_path, allowed_mem=100000)
    spec2 = cubed.Spec(tmp_path, allowed_mem=200000)
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec1)
    b = xp.ones((3, 3), chunks=(2, 2), spec=spec2)
    with pytest.raises(ValueError):
        xp.add(a, b)


class TestSpecMemArgTypes:
    def test_max_mem_deprecation_warning(self):
        # Remove once max_mem fully deprecated in favour of allowed_mem
        with pytest.warns(
            DeprecationWarning,
            match="`max_mem` is deprecated, please use `allowed_mem` instead",
        ):
            cubed.Spec(max_mem=100_000)

    @pytest.mark.parametrize(
        "input_value, expected_value",
        [
            (500, 500),
            (100_000, 100_000),
            ("500B", 500),
            ("1kB", 1000),
            ("1MB", 1000**2),
            ("1GB", 1000**3),
            ("1TB", 1000**4),
            ("1PB", 1000**5),
        ],
    )
    def test_convert_to_bytes(self, input_value, expected_value):
        spec = cubed.Spec(allowed_mem=input_value)
        assert spec.allowed_mem == expected_value

    @pytest.mark.parametrize(
        "input_value",
        [
            "1EB",  # EB is not a valid unit in this function
            "1kb",  # lower-case k is not valid
            "invalid",  # completely invalid input
            -512,  # negative integer
            1000.0,  # invalid type
        ],
    )
    def test_convert_to_bytes_error(self, input_value):
        with pytest.raises(ValueError):
            cubed.Spec(allowed_mem=input_value)


def test_reduction_multiple_rounds(tmp_path, executor):
    spec = cubed.Spec(tmp_path, allowed_mem=1000)
    a = xp.ones((100, 10), dtype=np.uint8, chunks=(1, 10), spec=spec)
    b = xp.sum(a, axis=0, dtype=np.uint8)
    # check that there is > 1 rechunk step
    rechunks = [
        n for (n, d) in b.plan.dag.nodes(data=True) if d["op_name"] == "rechunk"
    ]
    assert len(rechunks) > 1
    assert b.plan.max_projected_mem() == 1000
    assert_array_equal(b.compute(executor=executor), np.ones((100, 10)).sum(axis=0))


def test_reduction_not_enough_memory(tmp_path):
    spec = cubed.Spec(tmp_path, allowed_mem=50)
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
    spec1 = cubed.Spec(tmp_path, allowed_mem=100000)
    spec2 = cubed.Spec(tmp_path, allowed_mem=200000)

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
    # note we have to use dill which can serialize local functions, unlike pickle
    c = dill.loads(dill.dumps(c))

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

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(c.compute(), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]))


@pytest.mark.skipif(
    platform.system() == "Windows", reason="measuring memory does not run on windows"
)
def test_callbacks(spec, executor):
    if not isinstance(executor, DagExecutor):
        pytest.skip(f"{type(executor)} does not support callbacks")

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

    num_created_arrays = 3
    assert task_counter.value == num_created_arrays + 4


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

        num_created_arrays = 3
        assert task_counter.value == num_created_arrays + 4
    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


def test_already_computed(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)

    num_created_arrays = 4  # a, b, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 8

    task_counter = TaskCounter()
    c.compute(callbacks=[task_counter], optimize_graph=False)
    num_created_arrays = 3  # a, b, c
    assert task_counter.value == num_created_arrays + 4

    # since c has already been computed, when computing d only 4 tasks are run, instead of 8
    task_counter = TaskCounter()
    d.compute(callbacks=[task_counter], optimize_graph=False, resume=True)
    # the create arrays tasks are run again, even though they exist
    num_created_arrays = 4  # a, b, c, d
    assert task_counter.value == num_created_arrays + 4


@pytest.mark.skipif(platform.system() == "Windows", reason="does not run on windows")
def test_measure_reserved_mem(executor):
    pytest.importorskip("lithops")

    from cubed.runtime.executors.lithops import LithopsDagExecutor

    if not isinstance(executor, LithopsDagExecutor):
        pytest.skip(f"{type(executor)} does not support measure_reserved_mem")

    reserved_memory = cubed.measure_reserved_mem(executor=executor)
    assert reserved_memory > 1_000_000  # over 1MB


# Test we can create a plan for arrays of up to 5PB, and 100s of billions of tasks
@pytest.mark.parametrize("factor", [10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
def test_plan_scaling(tmp_path, factor):
    spec = cubed.Spec(tmp_path, allowed_mem="2GB")
    chunksize = 5000
    a = cubed.random.random(
        (factor * chunksize, factor * chunksize), chunks=chunksize, spec=spec
    )
    b = cubed.random.random(
        (factor * chunksize, factor * chunksize), chunks=chunksize, spec=spec
    )
    c = xp.matmul(a, b)

    assert c.plan.num_tasks() > 0
    c.visualize(filename=tmp_path / "c")
