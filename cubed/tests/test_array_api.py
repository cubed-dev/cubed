import fsspec
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed.array_api.manipulation_functions import reshape_chunks
from cubed.tests.utils import ALL_EXECUTORS, MODAL_EXECUTORS


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


@pytest.fixture(scope="module", params=ALL_EXECUTORS)
def executor(request):
    return request.param


@pytest.fixture(scope="module", params=MODAL_EXECUTORS)
def modal_executor(request):
    return request.param


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


def test_asarray_from_array(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray(a)
    assert b is a
    assert_array_equal(b.compute(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_eye(spec, k):
    a = xp.eye(5, k=k, chunks=(2, 2), spec=spec)
    assert_array_equal(a, np.eye(5, k=k))


def test_ones(spec, executor):
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec)
    assert_array_equal(a.compute(executor=executor), np.ones((3, 3)))


def test_ones_like(spec, executor):
    a = xp.ones((3, 3), chunks=(2, 2), spec=spec)
    b = xp.ones_like(a)
    assert_array_equal(b.compute(executor=executor), np.ones_like(np.ones((3, 3))))


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_tril_triu(spec, k):
    a = xp.ones((4, 5), chunks=(2, 2), spec=spec)
    assert_array_equal(xp.tril(a, k=k), np.tril(np.ones((4, 5)), k))
    assert_array_equal(xp.triu(a, k=k), np.triu(np.ones((4, 5)), k))


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
    a = xp.ones((10, 10), chunks=(10, 2), spec=spec)
    b = xp.ones((10, 10), chunks=(2, 10), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.ones((10, 10)) + np.ones((10, 10))
    )


def test_equal(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    c = xp.equal(a, b)
    assert_array_equal(c.compute(), np.full((3, 3), True))


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


@pytest.mark.cloud
def test_matmul_modal(modal_executor):
    tmp_path = "s3://cubed-unittest/matmul"
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
        assert_array_equal(c.compute(executor=modal_executor), expected)
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


def test_stack(spec, executor):
    a = xp.full((4, 6), 1, chunks=(2, 3), spec=spec)
    b = xp.full((4, 6), 2, chunks=(2, 3), spec=spec)
    c = xp.full((4, 6), 3, chunks=(2, 3), spec=spec)
    d = xp.stack([a, b, c], axis=0)
    assert_array_equal(
        d.compute(executor=executor),
        np.stack([np.full((4, 6), 1), np.full((4, 6), 2), np.full((4, 6), 3)], axis=0),
    )


# Searching functions


def test_argmax(spec):
    a = xp.asarray([[11, 12, 13], [11, 11, 14], [10, 13, 11]], chunks=(2, 2), spec=spec)
    b = xp.argmax(a)
    assert_array_equal(
        b.compute(),
        np.array([[11, 12, 13], [11, 11, 14], [10, 13, 11]]).argmax(),
    )


def test_argmax_axis_0(spec):
    a = xp.asarray([[11, 12, 13], [11, 11, 14], [10, 13, 11]], chunks=(2, 2), spec=spec)
    b = xp.argmax(a, axis=0)
    assert_array_equal(
        b.compute(),
        np.array([[11, 12, 13], [11, 11, 14], [10, 13, 11]]).argmax(axis=0),
    )


def test_argmin_axis_0(spec):
    a = xp.asarray([[11, 12, 13], [11, 11, 14], [10, 13, 11]], chunks=(2, 2), spec=spec)
    b = xp.argmin(a, axis=0)
    assert_array_equal(
        b.compute(),
        np.array([[11, 12, 13], [11, 11, 14], [10, 13, 11]]).argmin(axis=0),
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
