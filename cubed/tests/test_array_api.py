import fsspec
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cubed
import cubed.array_api as xp
from cubed.array_api.manipulation_functions import reshape_chunks
from cubed.tests.utils import ALL_EXECUTORS, MAIN_EXECUTORS, MODAL_EXECUTORS


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


def test_reflected_operation(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = 1 - a

    assert_array_equal(b.compute(), 1 - np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


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


@pytest.mark.parametrize("endpoint", [True, False])
def test_linspace(spec, endpoint):
    a = xp.linspace(6, 49, 50, endpoint=endpoint, chunks=5, spec=spec)
    npa = np.linspace(6, 49, 50, endpoint=endpoint)
    assert_allclose(a, npa)

    a = xp.linspace(1.4, 4.9, 13, endpoint=endpoint, chunks=5, spec=spec)
    npa = np.linspace(1.4, 4.9, 13, endpoint=endpoint)
    assert_allclose(a, npa)

    a = xp.linspace(0, 0, 0, endpoint=endpoint)
    npa = np.linspace(0, 0, 0, endpoint=endpoint)
    assert_allclose(a, npa)


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


# test on all executors (`any_executor`) for coverage
def test_add(spec, any_executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=any_executor), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
    )


def test_add_top_level_namespace(spec, executor):
    a = cubed.ones((10, 10), chunks=(10, 2), spec=spec)
    b = cubed.ones((10, 10), chunks=(2, 10), spec=spec)
    c = cubed.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.ones((10, 10)) + np.ones((10, 10))
    )


def test_add_different_chunks(spec, executor):
    a = xp.ones((10, 10), chunks=(10, 2), spec=spec)
    b = xp.ones((10, 10), chunks=(2, 10), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(executor=executor), np.ones((10, 10)) + np.ones((10, 10))
    )


def test_add_different_chunks_fail(spec, executor):
    a = xp.ones((10,), chunks=(3,), spec=spec)
    b = xp.ones((10,), chunks=(5,), spec=spec)
    c = xp.add(a, b)
    assert_array_equal(c.compute(executor=executor), np.ones((10,)) + np.ones((10,)))


def test_add_scalars():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))

    b = xp.add(a, 1)
    assert_array_equal(b.compute(), np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]))

    c = xp.add(2, a)
    assert_array_equal(c.compute(), np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]]))

    with pytest.raises(TypeError):
        xp.add(1, 2)


@pytest.mark.parametrize(
    "min, max",
    [
        (None, None),
        (4, None),
        (None, 7),
        (4, 7),
        (0, 10),
    ],
)
def test_clip(spec, min, max):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    npa = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = xp.clip(a, min, max)
    if min is max is None:
        assert b is a
    else:
        assert_array_equal(b.compute(), np.clip(npa, min, max))


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


@pytest.mark.parametrize(
    "ind",
    [
        Ellipsis,
        6,
        (6, xp.newaxis),
        (xp.newaxis, 6),
        slice(None),
        slice(10),
        slice(3, None),
        slice(3, 3),
        slice(3, 10),
        (slice(10), xp.newaxis),
    ],
)
def test_index_1d(spec, ind):
    a = xp.arange(12, chunks=(4,), spec=spec)
    assert_array_equal(a[ind].compute(), np.arange(12)[ind])


@pytest.mark.parametrize(
    "ind",
    [
        Ellipsis,
        (2, 3),
        (xp.newaxis, 2, 3),
        (Ellipsis, slice(2, 4)),
        (slice(None), slice(2, 2)),
        (slice(None), slice(2, 4)),
        (slice(3), slice(2, None)),
        (slice(1, None), slice(4)),
        (slice(1, 3), Ellipsis),
        (slice(1, 1), slice(None)),
        (slice(1, 3), slice(None)),
        (xp.newaxis, slice(None), slice(2, 4)),
        (slice(None), xp.newaxis, slice(2, 4)),
        (slice(None), slice(2, 4), xp.newaxis),
        (slice(None), 1),
        (1, slice(2, 4)),
    ],
)
def test_index_2d(spec, ind):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert_array_equal(a[ind].compute(), x[ind])


@pytest.mark.parametrize(
    "ind",
    [
        Ellipsis,
        (slice(None), slice(None)),
        (slice(0, 4), slice(None)),
        (slice(None), slice(0, 4)),
        (slice(0, 4), slice(0, 4)),
    ],
)
def test_index_2d_no_op(spec, ind):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    assert a is a[ind]


@pytest.mark.parametrize(
    "shape, chunks, ind, new_chunks_expected",
    [
        # step divides chunks exactly
        (20, 4, slice(3, 14, 2), ((4, 2),)),
        # step doesn't divide chunks exactly
        # chunks is prime (so merge_chunks can't restore chunks to 5)
        (20, 5, slice(3, 14, 2), ((4, 2),)),
        # step doesn't divide chunks exactly
        (20, 8, slice(5, 18, 3), ((5,),)),
        # step is bigger than chunks
        (50, 5, slice(3, 50, 7), ((5, 2),)),
    ],
)
def test_index_1d_step(spec, shape, chunks, ind, new_chunks_expected):
    a = xp.arange(shape, chunks=chunks, spec=spec)
    b = a[ind]
    assert_array_equal(b.compute(), np.arange(shape)[ind])
    assert b.chunks == new_chunks_expected


# fmt: off
@pytest.mark.parametrize(
    "shape, chunks, ind, new_chunks_expected",
    [
        (
            (20, 20),
            (4, 4),
            (slice(3, 14, 2), slice(3, 14, 3)),
            ((4, 2), (3, 1),),
        ),
    ],
)
# fmt: on
def test_index_2d_step(spec, shape, chunks, ind, new_chunks_expected):
    a = xp.ones(shape, chunks=chunks, spec=spec)
    b = a[ind]
    assert_array_equal(b.compute(), np.ones(shape)[ind])
    assert b.chunks == new_chunks_expected


def test_index_slice_unsupported_step(spec):
    with pytest.raises(NotImplementedError):
        a = xp.arange(12, chunks=(4,), spec=spec)
        a[::-1]


@pytest.mark.parametrize("axis", [0, 1])
def test_take(spec, axis):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray([1, 2], spec=spec)
    c = xp.take(a, b, axis=axis)
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    assert_array_equal(c.compute(), x.take([1, 2], axis=axis))


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
    spec = cubed.Spec(tmp_path, allowed_mem=100000)
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
    spec = cubed.Spec(tmp_path, allowed_mem=100000)
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
    c = xp.linalg.outer(a, b)
    assert_array_equal(c.compute(executor=executor), np.outer([0, 1, 2], [10, 50, 100]))


@pytest.mark.parametrize("axes", [1, (1, 0)])
def test_tensordot(axes):
    x = np.arange(400).reshape((20, 20))
    a = xp.asarray(x, chunks=(5, 4))
    y = np.arange(200).reshape((20, 10))
    b = xp.asarray(y, chunks=(4, 5))
    assert_array_equal(
        xp.tensordot(a, b, axes=axes).compute(), np.tensordot(x, y, axes=axes)
    )


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
        ((), (), (0,), None, ((0,),)),
        ((5, 1, 6), (3, 1, 3), (5, 0, 6), None, ((3, 2), (0,), (3, 3))),
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


def test_concat_different_chunks(spec):
    a = xp.asarray([[1], [5]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[2, 3, 4], [6, 7, 8]], chunks=(2, 3), spec=spec)
    c = xp.concat([a, b], axis=1)
    assert_array_equal(
        c.compute(),
        np.concatenate(
            [
                np.array([[1], [5]]),
                np.array([[2, 3, 4], [6, 7, 8]]),
            ],
            axis=1,
        ),
    )


@pytest.mark.parametrize("axis", [None, 0])
def test_concat_single_array(spec, axis):
    a = xp.full((4, 5), 1, chunks=(3, 2), spec=spec)
    d = xp.concat([a], axis=axis)
    assert_array_equal(
        d.compute(),
        np.concatenate([np.full((4, 5), 1)], axis=axis),
    )


def test_concat_incompatible_shapes(spec):
    a = xp.full((4, 5), 1, chunks=(3, 2), spec=spec)
    b = xp.full((4, 6), 2, chunks=(3, 2), spec=spec)
    with pytest.raises(
        ValueError,
        match="all the input array dimensions except for the concatenation axis must match exactly",
    ):
        xp.concat([a, b], axis=0)
    xp.concat([a, b], axis=1)  # OK


def test_expand_dims(spec, executor):
    a = xp.asarray([1, 2, 3], chunks=(2,), spec=spec)
    b = xp.expand_dims(a, axis=0)
    assert_array_equal(b.compute(executor=executor), np.expand_dims([1, 2, 3], 0))


@pytest.mark.parametrize(
    "shape, chunks, axis",
    [
        ((10,), (4,), None),
        ((10,), (4,), 0),
        ((10,), (5,), 0),
        ((10, 7), (4, 3), None),
        ((10, 7), (4, 3), 0),
        ((10, 7), (4, 3), 1),
        ((10, 7), (4, 3), (0, 1)),
        ((10, 7), (4, 3), -1),
        ((10, 7), (5, 3), (0, 1)),
    ],
)
def test_flip(executor, shape, chunks, axis):
    x = np.random.randint(10, size=shape)
    a = xp.asarray(x, chunks=chunks)
    b = xp.flip(a, axis=axis)

    assert b.chunks == a.chunks

    assert_array_equal(
        b.compute(executor=executor),
        np.flip(x, axis=axis),
    )


def test_moveaxis(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.moveaxis(a, [0, -1], [-1, 0])
    assert_array_equal(
        b.compute(),
        np.moveaxis(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [0, -1], [-1, 0]),
    )


def test_permute_dims(spec, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.permute_dims(a, (1, 0))
    assert_array_equal(
        b.compute(executor=executor),
        np.transpose(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
    )


def test_repeat(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.repeat(a, 3, axis=1)
    assert_array_equal(
        b.compute(),
        np.repeat(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 3, axis=1),
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


def test_reshape_chunks_with_smaller_end_chunk(spec, executor):
    a = xp.arange(10, chunks=4, spec=spec)
    b = reshape_chunks(a, (2, 5), (2, 2))

    assert b.shape == (2, 5)
    assert b.chunks == ((2,), (2, 2, 1))

    assert_array_equal(
        b.compute(executor=executor),
        np.array([[0, 1, 4, 5, 8], [2, 3, 6, 7, 9]]),
    )


def _maybe_len(a):
    try:
        return len(a)
    except TypeError:
        return 0


@pytest.mark.parametrize(
    "chunks, shift, axis",
    [
        ((2, 6), 3, None),
        ((2, 6), 3, 0),
        ((2, 6), (3, 9), (0, 1)),
        ((2, 6), (3, 9), None),
        ((2, 6), (3, 9), 1),
    ],
)
def test_roll(spec, executor, chunks, shift, axis):
    x = np.arange(4 * 6).reshape((4, 6))
    a = cubed.from_array(x, chunks=chunks, spec=spec)

    if _maybe_len(shift) != _maybe_len(axis):
        with pytest.raises(TypeError if axis is None else ValueError):
            xp.roll(a, shift, axis=axis)
    else:
        assert_array_equal(
            xp.roll(a, shift, axis=axis).compute(executor=executor),
            np.roll(x, shift, axis),
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


@pytest.mark.parametrize("repetitions", [(2,), (2, 5), (2, 5, 3)])
def test_tile(spec, repetitions):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.tile(a, repetitions)
    assert_array_equal(
        b.compute(),
        np.tile(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), repetitions),
    )


@pytest.mark.parametrize("chunks", [(1, 2, 3), (2, 2, 3), (3, 2, 3)])
def test_unstack(spec, executor, chunks):
    a = xp.full((4, 6), 1, chunks=(2, 3), spec=spec)
    b = xp.full((4, 6), 2, chunks=(2, 3), spec=spec)
    c = xp.full((4, 6), 3, chunks=(2, 3), spec=spec)
    d = xp.stack([a, b, c], axis=0)

    d = d.rechunk(chunks)

    au, bu, cu = cubed.compute(*xp.unstack(d), executor=executor, optimize_graph=False)

    assert_array_equal(au, np.full((4, 6), 1))
    assert_array_equal(bu, np.full((4, 6), 2))
    assert_array_equal(cu, np.full((4, 6), 3))


def test_unstack_zero_arrays(spec):
    a = xp.full((0, 4, 6), 1, chunks=(1, 2, 3), spec=spec)
    assert xp.unstack(a) == ()


def test_unstack_single_array(spec):
    a = xp.full((1, 4, 6), 1, chunks=(1, 2, 3), spec=spec)
    (b,) = xp.unstack(a)
    assert_array_equal(b.compute(), np.full((4, 6), 1))


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


@pytest.mark.parametrize(
    "x1, x1_chunks, x2, x2_chunks",
    [
        [[], 1, [], 1],
        [[0], 1, [0], 1],
        [[-10, 0, 10, 20, 30], 3, [11, 30], 2],
        [[-10, 0, 10, 20, 30], 3, [11, 30, -20, 1, -10, 10, 37, 11], 5],
        [[-10, 0, 10, 20, 30], 3, [[11, 30, -20, 1, -10, 10, 37, 11]], 5],
        [[-10, 0, 10, 20, 30], 3, [[7, 0], [-10, 10], [11, -1], [15, 15]], (2, 2)],
    ],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted(x1, x1_chunks, x2, x2_chunks, side):
    x1 = np.array(x1)
    x2 = np.array(x2)

    x1d = xp.asarray(x1, chunks=x1_chunks)
    x2d = xp.asarray(x2, chunks=x2_chunks)

    out = xp.searchsorted(x1d, x2d, side=side)

    assert out.shape == x2d.shape
    assert out.chunks == x2d.chunks
    assert_array_equal(out.compute(), np.searchsorted(x1, x2, side=side))


def test_searchsorted_sorter_not_implemented():
    with pytest.raises(NotImplementedError):
        xp.searchsorted(xp.asarray([1, 0]), xp.asarray([1]), sorter=xp.asarray([1, 0]))


def test_where_scalars():
    condition = xp.asarray(
        [[True, False, True], [False, True, False], [True, False, True]], chunks=(2, 2)
    )
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))

    b = xp.where(condition, a, 0)
    assert_array_equal(b.compute(), np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]]))

    c = xp.where(condition, 0, a)
    assert_array_equal(c.compute(), np.array([[0, 2, 0], [4, 0, 6], [0, 8, 0]]))

    with pytest.raises(TypeError):
        xp.where(condition, 0, 1)


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


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("correction", [0.0, 1.0])
@pytest.mark.parametrize("keepdims", [False, True])
def test_var(spec, axis, correction, keepdims):
    a = xp.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], chunks=(2, 2), spec=spec
    )
    b = xp.var(a, axis=axis, correction=correction, keepdims=keepdims)
    assert_array_equal(
        b.compute(optimize_graph=False),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).var(
            axis=axis, ddof=correction, keepdims=keepdims
        ),
    )


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("correction", [0.0, 1.0])
@pytest.mark.parametrize("keepdims", [False, True])
def test_std(spec, axis, correction, keepdims):
    a = xp.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], chunks=(2, 2), spec=spec
    )
    b = xp.std(a, axis=axis, correction=correction, keepdims=keepdims)
    assert_array_equal(
        b.compute(),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).std(
            axis=axis, ddof=correction, keepdims=keepdims
        ),
    )


def test_var__poorly_conditioned(spec):
    # from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Example
    npa = np.array([4.0, 7.0, 13.0, 16.0]) + 1e9
    a = xp.asarray(npa, chunks=2, spec=spec)
    b = xp.var(a, axis=0)
    assert_array_equal(b.compute(), npa.var(axis=0))


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
