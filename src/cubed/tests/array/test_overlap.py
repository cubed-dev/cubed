import numpy as np
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp


def test_map_overlap_1d():
    x = np.arange(6)
    a = xp.asarray(x, chunks=(3,))

    b = cubed.map_overlap(
        lambda x: x,
        a,
        dtype=a.dtype,
        chunks=((5, 5),),
        depth=1,
        boundary=0,
        trim=False,
    )

    assert_array_equal(b.compute(), np.array([0, 0, 1, 2, 3, 2, 3, 4, 5, 0]))


def test_map_overlap_1d_single_chunk():
    x = np.arange(6)
    a = xp.asarray(x, chunks=(6,))

    b = cubed.map_overlap(
        lambda x: x,
        a,
        dtype=a.dtype,
        chunks=((8,),),
        depth=1,
        boundary=0,
        trim=False,
    )

    assert_array_equal(b.compute(), np.array([0, 0, 1, 2, 3, 4, 5, 0]))


def test_map_overlap_1d_change_dtype():
    x = np.arange(6)
    a = xp.asarray(x, chunks=(3,))

    b = cubed.map_overlap(
        lambda x: x.astype(np.float64),
        a,
        dtype=np.float64,
        chunks=((5, 5),),
        depth=1,
        boundary=0,
        trim=False,
    )

    assert b.dtype == np.float64
    assert_array_equal(b.compute(), np.array([0, 0, 1, 2, 3, 2, 3, 4, 5, 0]))


def test_map_overlap_2d():
    x = np.arange(36).reshape((6, 6))
    a = xp.asarray(x, chunks=(3, 3))

    b = cubed.map_overlap(
        lambda x: x,
        a,
        dtype=a.dtype,
        chunks=((7, 7), (5, 5)),
        depth={0: 2, 1: 1},
        boundary={0: 100, 1: 200},
        trim=False,
    )

    expected = np.array(
        [
            [200, 100, 100, 100, 100, 100, 100, 100, 100, 200],
            [200, 100, 100, 100, 100, 100, 100, 100, 100, 200],
            [200, 0, 1, 2, 3, 2, 3, 4, 5, 200],
            [200, 6, 7, 8, 9, 8, 9, 10, 11, 200],
            [200, 12, 13, 14, 15, 14, 15, 16, 17, 200],
            [200, 18, 19, 20, 21, 20, 21, 22, 23, 200],
            [200, 24, 25, 26, 27, 26, 27, 28, 29, 200],
            [200, 6, 7, 8, 9, 8, 9, 10, 11, 200],
            [200, 12, 13, 14, 15, 14, 15, 16, 17, 200],
            [200, 18, 19, 20, 21, 20, 21, 22, 23, 200],
            [200, 24, 25, 26, 27, 26, 27, 28, 29, 200],
            [200, 30, 31, 32, 33, 32, 33, 34, 35, 200],
            [200, 100, 100, 100, 100, 100, 100, 100, 100, 200],
            [200, 100, 100, 100, 100, 100, 100, 100, 100, 200],
        ]
    )

    assert_array_equal(b.compute(), expected)


def test_map_overlap_trim():
    x = np.array([1, 1, 2, 3, 5, 8, 13, 21])
    a = xp.asarray(x, chunks=5)

    def derivative(x):
        out = x - np.roll(x, 1)
        return out[1:-1]  # manual trim

    b = cubed.map_overlap(
        derivative,
        a,
        dtype=a.dtype,
        chunks=a.chunks,
        depth=1,
        boundary=0,
        trim=False,
    )

    assert_array_equal(b.compute(), np.array([1, 0, 1, 1, 2, 3, 5, 8]))
