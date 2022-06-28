import pickle

import fsspec
import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed import Callback
from cubed.array_api.manipulation_functions import reshape_chunks
from cubed.primitive.blockwise import apply_blockwise
from cubed.runtime.executors.lithops import LithopsDagExecutor
from cubed.runtime.executors.python import PythonDagExecutor
from cubed.tests.utils import ALL_EXECUTORS, LITHOPS_LOCAL_CONFIG, create_zarr


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


@pytest.fixture(scope="module", params=ALL_EXECUTORS)
def executor(request):
    return request.param


# Test API

# Array object


def test_object_bool(tmp_path, executor):
    spec = cubed.Spec(tmp_path, 100000, executor=executor)
    a = xp.asarray(
        [[False, False, False], [False, False, False], [False, False, False]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.all(a)
    assert not b

    a = xp.asarray(
        [[True, True, True], [True, True, True], [True, True, True]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.all(a)
    assert b


def test_object_transpose(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)

    assert_array_equal(a.mT.compute(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T)

    assert_array_equal(a.T.compute(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T)


# Creation functions


def test_arange(spec, executor):
    a = xp.arange(12, chunks=(5,), spec=spec)
    assert_array_equal(a.compute(executor=executor), np.arange(12))


def test_arange_step(spec):
    a = xp.arange(20, step=3, chunks=(5,), spec=spec)
    assert_array_equal(a.compute(), np.arange(20, step=3))


def test_asarray(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    assert_array_equal(
        a.compute(executor=executor), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )


def test_ones(spec, executor):
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec)
    assert_array_equal(a.compute(executor=executor), np.ones((3, 3)))


def test_ones_like(spec, executor):
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec)
    b = xp.ones_like(a)
    assert_array_equal(b.compute(executor=executor), np.ones_like(np.ones((3, 3))))


# Data type functions


def test_astype(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.astype(a, xp.int32)
    assert_array_equal(
        b.compute(executor=executor),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )


# Elementwise functions


def test_add(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    )


def test_add_with_broadcast(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([1, 1, 1], chunks=(2,), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    )


def test_equal(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    c = xp.equal(a, b)
    assert_array_equal(c.compute(executor=executor), np.full((3, 3), True))


def test_negative(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    assert_array_equal(
        b.compute(executor=executor),
        np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]),
    )


# Indexing


@pytest.mark.parametrize("i", [6])
def test_index_1d(spec, i):
    a = xp.arange(12, chunks=(4,), spec=spec)
    assert_array_equal(a[i].compute(), np.arange(12)[i])


@pytest.mark.parametrize("i", [(2, 3)])
def test_index_2d(spec, i):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert_array_equal(a[i].compute(), x[i])


@pytest.mark.parametrize("sl", [slice(None), slice(10), slice(3, None), slice(3, 10)])
def test_slice_1d(spec, sl):
    a = xp.arange(12, chunks=(4,), spec=spec)
    assert_array_equal(a[sl].compute(), np.arange(12)[sl])


@pytest.mark.parametrize(
    "sl0, sl1",
    [
        (slice(None), slice(2, 4)),
        (slice(3), slice(2, None)),
        (slice(1, None), slice(4)),
        (slice(1, 3), slice(None)),
    ],
)
def test_slice_2d(spec, sl0, sl1):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert_array_equal(a[sl0, sl1].compute(), x[sl0, sl1])


def test_slice_unsupported_step(spec):
    with pytest.raises(NotImplementedError):
        a = xp.arange(12, chunks=(4,), spec=spec)
        a[3:10:2]


def test_setitem(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.ones(())
    a[1, 2] = b
    assert_array_equal(a.compute(), np.array([[1, 2, 3], [4, 5, 1], [7, 8, 9]]))


def test_setitem_fails_not_0d(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[4, 5, 6], [7, 8, 9], [1, 2, 3]], chunks=(2, 2), spec=spec)
    with pytest.raises(NotImplementedError):
        a[:] = b


def test_setitem_fails_not_computed(spec):
    a = xp.arange(12, chunks=(4,), spec=spec)
    b = xp.ones(())
    with pytest.raises(NotImplementedError):
        a[1] = b


# Linear algebra functions


def test_matmul(spec, executor):
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
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    y = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    expected = np.matmul(x, y)
    assert_array_equal(c.compute(executor=executor), expected)


@pytest.mark.cloud
def test_matmul_cloud(executor):
    tmp_path = "gs://barry-zarr-test/matmul"
    spec = cubed.Spec(tmp_path, max_mem=100000)
    try:
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
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        y = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        expected = np.matmul(x, y)
        assert_array_equal(c.compute(executor=executor), expected)
    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


def test_outer(spec, executor):
    a = xp.asarray([0, 1, 2], chunks=2, spec=spec)
    b = xp.asarray([10, 50, 100], chunks=2, spec=spec)
    c = xp.outer(a, b)
    assert_array_equal(c.compute(executor=executor), np.outer([0, 1, 2], [10, 50, 100]))


# Manipulation functions


def test_broadcast_arrays(executor):
    a = xp.ones(30, chunks=(3,))
    b = xp.ones(30, chunks=(6,))
    a_b, b_b = xp.broadcast_arrays(a, b)

    assert_array_equal(a_b.compute(), np.ones(30))
    assert_array_equal(b_b.compute(), np.ones(30))

    a = xp.ones((1, 30), chunks=(1, 3))
    b = xp.ones(30, chunks=(6,))
    a_b, b_b = xp.broadcast_arrays(a, b)

    assert_array_equal(a_b.compute(executor=executor), np.ones((1, 30)))
    assert_array_equal(b_b.compute(executor=executor), np.ones((1, 30)))


@pytest.mark.parametrize(
    "shape, chunks, new_shape, new_chunks, new_chunks_expected",
    [
        # ((5, 1, 6), (3, 1, 3), (5, 0, 6), None, ((3, 2), (0,), (3, 3))),  # fails
        ((5, 1, 6), (3, 1, 3), (5, 4, 6), None, ((3, 2), (1, 1, 1, 1), (3, 3))),
        ((5, 1, 6), (3, 1, 3), (2, 5, 1, 6), None, ((1, 1), (3, 2), (1,), (3, 3))),
        ((5, 1, 6), (3, 1, 3), (5, 3, 6), (3, 3, 3), ((3, 2), (3,), (3, 3))),
    ],
)
def test_broadcast_to(
    executor, shape, chunks, new_shape, new_chunks, new_chunks_expected
):
    x = np.random.randint(10, size=shape)
    a = xp.asarray(x, chunks=chunks)
    b = xp.broadcast_to(a, shape=new_shape, chunks=new_chunks)

    assert b.shape == new_shape
    assert b.chunks == new_chunks_expected
    assert_array_equal(b.compute(executor=executor), np.broadcast_to(x, new_shape))


def test_concat(spec, executor):
    # note: middle chunk of output reads from three input chunks
    a = xp.full((4, 5), 1, chunks=(3, 2), spec=spec)
    b = xp.full((1, 5), 2, chunks=(3, 2), spec=spec)
    c = xp.full((3, 5), 3, chunks=(3, 2), spec=spec)
    d = xp.concat([a, b, c], axis=0)
    assert_array_equal(
        d.compute(executor=executor),
        np.concatenate(
            [np.full((4, 5), 1), np.full((1, 5), 2), np.full((3, 5), 3)], axis=0
        ),
    )


def test_expand_dims(spec, executor):
    a = xp.asarray([1, 2, 3], chunks=(2,), spec=spec)
    b = xp.expand_dims(a, axis=0)
    assert_array_equal(b.compute(executor=executor), np.expand_dims([1, 2, 3], 0))


def test_permute_dims(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.permute_dims(a, (1, 0))
    assert_array_equal(
        b.compute(executor=executor),
        np.transpose(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
    )


def test_reshape(spec, executor):
    a = xp.arange(12, chunks=4, spec=spec)
    b = xp.reshape(a, (3, 4))

    assert_array_equal(
        b.compute(executor=executor),
        np.arange(12).reshape((3, 4)),
    )


def test_reshape_chunks(spec, executor):
    a = xp.arange(12, chunks=4, spec=spec)
    b = reshape_chunks(a, (2, 6), (2, 2))

    assert b.shape == (2, 6)
    assert b.chunks == ((2,), (2, 2, 2))

    assert_array_equal(
        b.compute(executor=executor),
        np.array([[0, 1, 4, 5, 8, 9], [2, 3, 6, 7, 10, 11]]),
    )


def test_stack(spec, executor):
    a = xp.full((4, 6), 1, chunks=(2, 3), spec=spec)
    b = xp.full((4, 6), 2, chunks=(2, 3), spec=spec)
    c = xp.full((4, 6), 3, chunks=(2, 3), spec=spec)
    d = xp.stack([a, b, c], axis=0)
    assert_array_equal(
        d.compute(executor=executor),
        np.stack([np.full((4, 6), 1), np.full((4, 6), 2), np.full((4, 6), 3)], axis=0),
    )


def test_squeeze_1d(spec, executor):
    a = xp.asarray([[1, 2, 3]], chunks=(1, 2), spec=spec)
    b = xp.squeeze(a, 0)
    assert_array_equal(b.compute(executor=executor), np.squeeze([[1, 2, 3]], 0))


def test_squeeze_2d(spec, executor):
    a = xp.asarray([[[1], [2], [3]]], chunks=(1, 2, 1), spec=spec)
    b = xp.squeeze(a, (0, 2))
    assert_array_equal(
        b.compute(executor=executor), np.squeeze([[[1], [2], [3]]], (0, 2))
    )


# Statistical functions


def test_mean_axis_0(spec, executor):
    a = xp.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], chunks=(2, 2), spec=spec
    )
    b = xp.mean(a, axis=0)
    assert_array_equal(
        b.compute(executor=executor),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).mean(axis=0),
    )


def test_sum(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.sum(a)
    assert_array_equal(
        b.compute(executor=executor), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).sum()
    )


def test_sum_axis_0(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.sum(a, axis=0)
    assert_array_equal(b.compute(executor=executor), np.array([12, 15, 18]))


# Utility functions


def test_all(spec, executor):
    a = xp.asarray(
        [[True, True, True], [True, True, True], [True, True, True]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.all(a)
    assert b.ndim == 0
    assert b.size == 1
    assert b.compute(executor=executor)


def test_all_zero_dimension(spec, executor):
    a = xp.ones((0,), spec=spec)
    b = xp.all(a)
    assert b.ndim == 0
    assert b.size == 1
    assert b.compute(executor=executor)


# Other


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


def test_unify_chunks(spec, executor):
    a = xp.ones((10, 10), chunks=(10, 2), spec=spec)
    b = xp.ones((10, 10), chunks=(2, 10), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.ones((10, 10)) + np.ones((10, 10))
    )


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


def test_retries_lithops(mocker, spec):
    # Inject faults into the primitive layer
    # We need to use random faults, since we can't coordinate using object state
    def random_failure_apply_blockwise(*args, **kwargs):
        import random

        if random.random() < 0.2:
            raise IOError("Test fault injection")
        return apply_blockwise(*args, **kwargs)

    mocker.patch(
        "cubed.primitive.blockwise.apply_blockwise",
        side_effect=random_failure_apply_blockwise,
    )

    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
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

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor, callbacks=[task_counter]),
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
