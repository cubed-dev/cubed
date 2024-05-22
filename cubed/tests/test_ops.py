import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed.array_api as xp
from cubed.core.ops import smallest_blockdim, unify_chunks
from cubed.tests.utils import TaskCounter


def test_smallest_blockdim():
    assert smallest_blockdim([]) == ()
    assert smallest_blockdim([(5,), (5,)]) == (5,)
    assert smallest_blockdim([(5,), (3, 2)]) == (3, 2)
    assert smallest_blockdim([(5, 5), (3, 3, 3, 1)]) == (3, 3, 3, 1)
    assert smallest_blockdim([(2, 1), (2, 1)]) == (2, 1)
    assert smallest_blockdim([(2, 2, 1), (3, 2), (2, 2, 1)]) == (2, 2, 1)

    with pytest.raises(ValueError, match="Chunks do not add up to same value"):
        smallest_blockdim([(2, 1), (2, 2)])


@pytest.mark.parametrize(
    "chunks_a, chunks_b, expected_chunksize",
    [
        ((2,), (4,), (2,)),
        ((4,), (2,), (2,)),
        ((6,), (10,), (6,)),
        ((10,), (10,), (10,)),
        ((5,), (10,), (5,)),
        ((3,), (5,), (3,)),
        ((5,), (3,), (3,)),
    ],
)
def test_unify_chunks_elemwise(chunks_a, chunks_b, expected_chunksize):
    a = xp.ones((10,), chunks=chunks_a)
    b = xp.ones((10,), chunks=chunks_b)

    _, arrays = unify_chunks(a, "i", b, "i")
    for arr in arrays:
        assert arr.chunksize == expected_chunksize

    c = xp.add(a, b)
    assert_array_equal(c.compute(), np.ones((10,)) + np.ones((10,)))


@pytest.mark.parametrize(
    "chunks_a, chunks_b, expected_chunksize",
    [
        ((2, 2), (4, 4), (2, 2)),
        ((2, 4), (4, 2), (2, 2)),
        ((4, 2), (2, 4), (2, 2)),
        ((3, 5), (5, 3), (3, 3)),
        ((3, 10), (10, 3), (3, 3)),
    ],
)
def test_unify_chunks_elemwise_2d(chunks_a, chunks_b, expected_chunksize):
    a = xp.ones((10, 10), chunks=chunks_a)
    b = xp.ones((10, 10), chunks=chunks_b)

    _, arrays = unify_chunks(a, "ij", b, "ij")
    for arr in arrays:
        assert arr.chunksize == expected_chunksize

    c = xp.add(a, b)
    assert_array_equal(c.compute(), np.ones((10, 10)) + np.ones((10, 10)))


@pytest.mark.parametrize(
    "chunks_a, chunks_b, expected_chunksize",
    [
        ((2, 2), (4, 4), (2, 2)),
        ((2, 4), (2, 4), (2, 2)),
        ((4, 2), (4, 2), (2, 2)),
        ((3, 5), (3, 5), (3, 3)),
        ((3, 10), (3, 10), (3, 3)),
    ],
)
def test_unify_chunks_blockwise_2d(chunks_a, chunks_b, expected_chunksize):
    a = xp.ones((10, 10), chunks=chunks_a)
    b = xp.ones((10, 10), chunks=chunks_b)

    _, arrays = unify_chunks(a, "ij", b, "ji")
    for arr in arrays:
        assert arr.chunksize == expected_chunksize

    c = xp.matmul(a, b)
    assert_array_equal(c.compute(), np.matmul(np.ones((10, 10)), np.ones((10, 10))))


def test_unify_chunks_broadcast_scalar():
    a = xp.ones((10,), chunks=(3,))
    b = a + 1
    assert_array_equal(b.compute(), np.ones((10,)) + 1)


def test_unify_chunks_broadcast_2d():
    a = xp.ones((10, 10), chunks=(3, 3))
    b = xp.ones((10,), chunks=(5,))
    c = xp.add(a, b)

    # the following checks that b is rechunked *before* broadcasting, to avoid materializing the full (broadcasted) array
    task_counter = TaskCounter()
    res = c.compute(callbacks=[task_counter])
    num_created_arrays = 2  # b rechunked, c
    # 1 task for rechunk of b, 16 for addition operation
    assert task_counter.value == num_created_arrays + 1 + 16

    assert_array_equal(res, np.ones((10, 10)) + np.ones((10,)))
