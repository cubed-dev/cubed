import platform
import random
from functools import partial

import dill
import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.array_api.dtypes import _floating_dtypes
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import general_blockwise, merge_chunks, partial_reduce, tree_reduce
from cubed.core.optimization import fuse_all_optimize_dag, multiple_inputs_optimize_dag
from cubed.storage.backend import open_backend_array
from cubed.tests.utils import ALL_EXECUTORS, MAIN_EXECUTORS, TaskCounter, create_zarr


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


def test_as_array_fails(spec):
    a = np.ones((1000, 1000))
    with pytest.raises(
        ValueError,
        match="Size of in memory array is 8.0 MB which exceeds maximum of 1.0 MB.",
    ):
        xp.asarray(a, chunks=(100, 100), spec=spec)


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
    assert_array_equal(a, za[:])


@pytest.mark.parametrize("path", [None, "sub", "sub/group"])
def test_from_zarr(tmp_path, spec, executor, path):
    store = store = tmp_path / "source.zarr"
    create_zarr(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        store=store,
        path=path,
    )
    a = cubed.from_zarr(store, path=path, spec=spec)
    assert_array_equal(
        a.compute(executor=executor), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )


def test_store(tmp_path, spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)

    store = tmp_path / "source.zarr"
    target = zarr.empty(a.shape, chunks=a.chunksize, store=store)

    cubed.store(a, target, executor=executor)
    assert_array_equal(target[:], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_store_multiple(tmp_path, spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)

    store1 = tmp_path / "source1.zarr"
    target1 = zarr.empty(a.shape, chunks=a.chunksize, store=store1)
    store2 = tmp_path / "source2.zarr"
    target2 = zarr.empty(b.shape, chunks=b.chunksize, store=store2)

    cubed.store([a, b], [target1, target2], executor=executor)
    assert_array_equal(target1[:], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert_array_equal(target2[:], np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))


def test_store_fails(tmp_path, spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    store = tmp_path / "source.zarr"
    target = zarr.empty(a.shape, chunks=a.chunksize, store=store)

    with pytest.raises(
        ValueError, match=r"Different number of sources \(2\) and targets \(1\)"
    ):
        cubed.store([a, b], [target], executor=executor)

    with pytest.raises(ValueError, match="All sources must be cubed array objects"):
        cubed.store([1], [target], executor=executor)


@pytest.mark.parametrize("path", [None, "sub", "sub/group"])
def test_to_zarr(tmp_path, spec, executor, path):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    store = tmp_path / "output.zarr"
    cubed.to_zarr(a, store, path=path, executor=executor)
    res = open_backend_array(store, mode="r", path=path)
    assert_array_equal(res[:], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_map_blocks_with_kwargs(spec, executor):
    # based on dask test
    a = xp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], chunks=5, spec=spec)
    b = cubed.map_blocks(nxp.max, a, axis=0, keepdims=True, dtype=a.dtype, chunks=(1,))
    assert_array_equal(b.compute(executor=executor), np.array([4, 9]))


def test_map_blocks_with_block_id(spec, executor):
    # based on dask test
    def func(block, block_id=None, c=0):
        return nxp.ones_like(block) * int(sum(block_id)) + c

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


def test_map_blocks_no_array_args(spec, executor):
    def func(block, block_id=None):
        return nxp.ones_like(block) * int(sum(block_id))

    a = cubed.map_blocks(func, dtype="int64", chunks=((5, 3),), spec=spec)
    assert a.chunks == ((5, 3),)

    assert_array_equal(
        a.compute(executor=executor),
        np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype="int64"),
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


def test_map_blocks_drop_axis_chunking(spec):
    # This tests the case illustrated in https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html
    # Unlike Dask, Cubed does not support concatenating chunks, and will fail if the dropped axis has multiple chunks.

    def func(x):
        return nxp.sum(x, axis=2)

    an = np.arange(8 * 6 * 2).reshape((8, 6, 2))

    # single chunk in axis=2 works fine
    a = xp.asarray(an, chunks=(5, 4, 2), spec=spec)
    b = cubed.map_blocks(func, a, drop_axis=2)
    assert_array_equal(b.compute(), np.sum(an, axis=2))

    # multiple chunks in axis=2 raises
    a = xp.asarray(an, chunks=(5, 4, 1), spec=spec)
    with pytest.raises(
        ValueError, match=r"Cannot have multiple chunks in dropped axis 2."
    ):
        cubed.map_blocks(func, a, drop_axis=2)


def test_map_blocks_with_non_cubed_array(spec):
    a = xp.arange(10, dtype="int64", chunks=(2,), spec=spec)
    b = np.array([1, 2], dtype="int64")  # numpy array will be coerced to cubed
    c = cubed.map_blocks(nxp.add, a, b, dtype="int64")
    assert_array_equal(c.compute(), np.array([1, 3, 3, 5, 5, 7, 7, 9, 9, 11]))


def test_multiple_ops(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    d = xp.negative(c)
    assert_array_equal(
        d.compute(executor=executor),
        np.array([[-2, -3, -4], [-5, -6, -7], [-8, -9, -10]]),
    )


@pytest.mark.parametrize(
    ("new_chunks", "expected_chunks"),
    [
        ((1, 2), ((1, 1, 1), (2, 1))),
        ({0: 1, 1: 2}, ((1, 1, 1), (2, 1))),
        ({1: 2}, ((2, 1), (2, 1))),  # dim 0 unchanged
        ({}, ((2, 1), (1, 1, 1))),  # unchanged
    ],
)
def test_rechunk(spec, executor, new_chunks, expected_chunks):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 1), spec=spec)
    b = a.rechunk(new_chunks)
    assert b.chunks == expected_chunks
    assert_array_equal(
        b.compute(executor=executor),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )


def test_rechunk_same_chunks(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 1), spec=spec)
    b = a.rechunk((2, 1))
    assert b is a
    task_counter = TaskCounter()
    res = b.compute(callbacks=[task_counter])
    # no tasks should have run since chunks are same
    assert task_counter.value == 0

    assert_array_equal(res, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


# see also test_rechunk.py
def test_rechunk_intermediate(tmp_path):
    # factor of 4 is for chunks copies, extra 8 is for map_selection
    spec = cubed.Spec(tmp_path, allowed_mem=5 * 8 * 4 + 8)
    a = xp.ones((5, 5), chunks=(1, 5), spec=spec)
    b = a.rechunk((5, 1))
    assert_array_equal(b.compute(), np.ones((5, 5)))
    # intermediates = [n for (n, d) in b.plan.dag.nodes(data=True) if "-int" in d["name"]]
    # assert len(intermediates) == 1
    rechunks = [
        n
        for (n, d) in b.plan.dag.nodes(data=True)
        if d.get("op_name", None) == "rechunk"
    ]
    assert len(rechunks) == 2  # two ops due to intermediate store


def test_rechunk_merge_chunks_optimization():
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 1),
    )
    b = a.rechunk((4, 2))
    assert b.chunks == ((4,), (2, 2))
    assert_array_equal(
        b.compute(),
        np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
    )
    rechunks = [
        n
        for (n, d) in b.plan.dag.nodes(data=True)
        if d.get("op_name", None) == "rechunk"
    ]
    assert len(rechunks) == 0


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
    a = xp.ones((20000, 10000), chunks=(10000, 10000))
    with pytest.raises(ValueError):
        xp.negative(a)


def test_default_spec_config_override():
    # override default spec to increase allowed_mem
    from cubed import config

    with config.set(
        {"spec.allowed_mem": "4GB", "spec.executor_name": "single-threaded"}
    ):
        a = xp.ones((20000, 10000), chunks=(10000, 10000))
        b = xp.negative(a)
        assert_array_equal(b.compute(), -np.ones((20000, 10000)))


@pytest.mark.parametrize(
    "compressor",
    [
        None,
        {"id": "zstd", "level": 1},
        {"id": "blosc", "cname": "lz4", "clevel": 2, "shuffle": -1},
    ],
)
def test_spec_compressor(tmp_path, compressor):
    spec = cubed.Spec(tmp_path, allowed_mem=100000, zarr_compressor=compressor)
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    assert_array_equal(b.compute(), -np.ones((3, 3)))


def test_different_specs(tmp_path):
    spec1 = cubed.Spec(tmp_path, allowed_mem=100000)
    spec2 = cubed.Spec(tmp_path, allowed_mem=200000)
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec1)
    b = xp.ones((3, 3), chunks=(2, 2), spec=spec2)
    with pytest.raises(ValueError):
        xp.add(a, b)


@pytest.mark.parametrize(
    "input_value, expected_value",
    [
        (500, 500),
        (100_000, 100_000),
        (50.0, 50),
        ("500B", 500),
        ("1kB", 1000),
        ("1MB", 1000**2),
        ("1GB", 1000**3),
        ("1TB", 1000**4),
        ("1PB", 1000**5),
        ("100_000", 100_000),
        ("1.2MB", 1.2 * 1000**2),
        ("1 MB", 1000**2),
        ("1.2 MB", 1.2 * 1000**2),
    ],
)
def test_convert_to_bytes(input_value, expected_value):
    spec = cubed.Spec(allowed_mem=input_value)
    assert spec.allowed_mem == expected_value


@pytest.mark.parametrize(
    "input_value",
    [
        "1EB",  # EB is not a valid unit in this function
        "1kb",  # lower-case k is not valid
        "invalid",  # completely invalid input
        -512,  # negative integer
        "kB",  # only unit, no value
        "1.1B",  # can't have a fractional number of bytes
    ],
)
def test_convert_to_bytes_error(input_value):
    with pytest.raises(ValueError):
        cubed.Spec(allowed_mem=input_value)


def test_reduction_multiple_rounds(tmp_path, executor):
    spec = cubed.Spec(tmp_path, allowed_mem=1000)
    a = xp.ones((100, 10), dtype=np.uint8, chunks=(1, 10), spec=spec)
    b = xp.sum(a, axis=0, dtype=np.uint8)
    # check that there is > 1 blockwise step (after optimization)
    finalized_plan = b.plan._finalize()
    blockwises = [
        n
        for (n, d) in finalized_plan.dag.nodes(data=True)
        if d.get("op_name", None) == "blockwise"
    ]
    assert len(blockwises) > 1
    assert finalized_plan.max_projected_mem() <= 1000
    assert_array_equal(b.compute(executor=executor), np.ones((100, 10)).sum(axis=0))


def test_partial_reduce(spec):
    a = xp.asarray(np.arange(242).reshape((11, 22)), chunks=(3, 4), spec=spec)
    b = partial_reduce(a, np.sum, split_every={0: 2})
    c = partial_reduce(b, np.sum, split_every={0: 2})
    assert_array_equal(
        c.compute(), np.arange(242).reshape((11, 22)).sum(axis=0, keepdims=True)
    )


def test_tree_reduce(spec):
    a = xp.asarray(np.arange(242).reshape((11, 22)), chunks=(3, 4), spec=spec)
    b = tree_reduce(a, np.sum, axis=0, dtype=np.int64, split_every={0: 2})
    assert_array_equal(
        b.compute(), np.arange(242).reshape((11, 22)).sum(axis=0, keepdims=True)
    )


@pytest.mark.parametrize(
    "target_chunks, expected_chunksize",
    [
        ((2, 3), None),
        ((4, 3), None),
        ((2, 6), None),
        ((4, 6), None),
        ((12, 12), (10, 10)),
    ],
)
def test_merge_chunks(spec, target_chunks, expected_chunksize):
    a = xp.ones((10, 10), dtype=np.uint8, chunks=(2, 3), spec=spec)
    b = merge_chunks(a, target_chunks)
    assert b.chunksize == (expected_chunksize or target_chunks)
    assert_array_equal(b.compute(), np.ones((10, 10)))


@pytest.mark.parametrize(
    "target_chunks", [(2,), (2, 3, 1), (3, 2), (1, 3), (5, 5), (10, 10)]
)
def test_merge_chunks_fails(spec, target_chunks):
    a = xp.ones((10, 10), dtype=np.uint8, chunks=(2, 3), spec=spec)
    with pytest.raises(ValueError):
        merge_chunks(a, target_chunks)


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
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=xp.float64, chunks=(2, 2))
    b = cubed.random.random((3, 3), chunks=(2, 2))
    c = xp.add(a, b)
    d = c.rechunk((3, 1))
    e = c * 3

    f = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    g = f * 4

    assert not (tmp_path / "e.dot").exists()
    assert not (tmp_path / "e.png").exists()
    assert not (tmp_path / "e.svg").exists()
    assert not (tmp_path / "dg.svg").exists()

    e.visualize(filename=tmp_path / "e")
    assert (tmp_path / "e.svg").exists()

    e.visualize(filename=tmp_path / "e-hidden", show_hidden=True)
    assert (tmp_path / "e-hidden.svg").exists()

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


@pytest.mark.skipif(platform.system() == "Windows", reason="does not run on windows")
def test_measure_reserved_mem(executor):
    if executor.name not in ("processes", "lithops"):
        pytest.skip(f"{executor.name} executor does not support measure_reserved_mem")

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

    assert c.plan._finalize().num_tasks() > 0
    c.visualize(filename=tmp_path / "c")


@pytest.mark.parametrize("t_length", [50, 500, 5000, 50000])
def test_plan_quad_means(tmp_path, t_length):
    # based on sizes from https://gist.github.com/TomNicholas/c6a28f7c22c6981f75bce280d3e28283
    spec = cubed.Spec(tmp_path, allowed_mem="2GB", reserved_mem="100MB")
    u = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1), spec=spec)
    v = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1), spec=spec)
    uv = u * v
    m = xp.mean(uv, axis=0, split_every=10)

    assert m.plan._finalize().num_tasks() > 0
    m.visualize(
        filename=tmp_path / "quad_means_unoptimized",
        optimize_graph=False,
        show_hidden=True,
    )
    m.visualize(
        filename=tmp_path / "quad_means",
        optimize_function=multiple_inputs_optimize_dag,
        show_hidden=True,
    )


def quad_means(tmp_path, t_length):
    # based on sizes from https://gist.github.com/TomNicholas/c6a28f7c22c6981f75bce280d3e28283
    spec = cubed.Spec(tmp_path, allowed_mem="2GB", reserved_mem="100MB")
    u = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1), spec=spec)
    v = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1), spec=spec)
    uv = u * v
    m = xp.mean(uv, axis=0)
    return m


def test_quad_means(tmp_path, t_length=50):
    # run twice, with and without optimization
    # set the random seed to ensure deterministic results
    random.seed(42)
    m0 = quad_means(tmp_path, t_length)

    random.seed(42)
    m1 = quad_means(tmp_path, t_length)

    m1.visualize(
        filename=tmp_path / "quad_means", optimize_function=fuse_all_optimize_dag
    )

    cubed.to_zarr(m0, store=tmp_path / "result0")
    cubed.to_zarr(
        m1, store=tmp_path / "result1", optimize_function=fuse_all_optimize_dag
    )

    res0 = open_backend_array(tmp_path / "result0", mode="r")
    res1 = open_backend_array(tmp_path / "result1", mode="r")

    assert_array_equal(res0[:], res1[:])


def test_quad_means_zarr(tmp_path, t_length=50):
    # write inputs to Zarr first to test more realistic usage pattern
    spec = cubed.Spec(tmp_path, allowed_mem="2GB", reserved_mem="100MB")
    u = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1), spec=spec)
    v = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1), spec=spec)

    arrays = [u, v]
    paths = [f"{tmp_path}/u_{t_length}.zarr", f"{tmp_path}/v_{t_length}.zarr"]
    cubed.store(arrays, paths)

    u = cubed.from_zarr(f"{tmp_path}/u_{t_length}.zarr", spec=spec)
    v = cubed.from_zarr(f"{tmp_path}/v_{t_length}.zarr", spec=spec)
    uv = u * v
    m = xp.mean(uv, axis=0, split_every=10)

    opt_fn = partial(multiple_inputs_optimize_dag, max_total_num_input_blocks=40)

    m.visualize(filename=tmp_path / "quad_means", optimize_function=opt_fn)

    cubed.to_zarr(m, store=tmp_path / "result", optimize_function=opt_fn)


def sqrts(x):
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrts")

    def _sqrts(x):
        yield nxp.sqrt(x)
        yield -nxp.sqrt(x)

    def block_function(out_key):
        return ((x.name,) + out_key[1:],)

    return general_blockwise(
        _sqrts,
        block_function,
        x,
        shapes=[x.shape, x.shape],
        dtypes=[x.dtype, x.dtype],
        chunkss=[x.chunks, x.chunks],
        target_stores=[None, None],  # filled in by general_blockwise
    )


def test_multiple_outputs():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), dtype=float)
    b, c = sqrts(a)

    cubed.compute(b, c)

    input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_array_equal(b, np.sqrt(input))
    assert_array_equal(c, -np.sqrt(input))
