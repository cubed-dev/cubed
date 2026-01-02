import numpy as np
import pytest

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal
from cubed.array.update import append, set2_
from cubed.storage.store import open_storage_array
from cubed.tests.utils import create_zarr


@pytest.mark.parametrize("axis", [0, 1])
def test_append(tmp_path, axis):
    store = tmp_path / "a.zarr"
    create_zarr(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(3, 3),
        store=store,
    )

    a = cubed.from_zarr(store, mode="r+")
    b = xp.full((3, 3), 2, chunks=(3, 3))
    c = append(a, b, axis=axis)

    c.compute(_return_in_memory_array=False)  # don't load into memory

    res = open_storage_array(store, mode="r")

    assert_array_equal(
        res[:],
        np.concatenate(
            [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.full((3, 3), 2)], axis=axis
        ),
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_append_uneven(tmp_path, axis):
    store = tmp_path / "a.zarr"
    create_zarr(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        store=store,
    )

    a = cubed.from_zarr(store, mode="r+")
    b = xp.full((3, 3), 2, chunks=(2, 2))
    c = append(a, b, axis=axis)

    c.visualize(optimize_graph=True)
    c.compute(_return_in_memory_array=False)  # don't load into memory

    res = open_storage_array(store, mode="r")

    assert_array_equal(
        res[:],
        np.concatenate(
            [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.full((3, 3), 2)], axis=axis
        ),
    )


def test_set_scalar(tmp_path):
    store = tmp_path / "a.zarr"
    create_zarr(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        store=store,
    )
    a = cubed.from_zarr(store, mode="r+")
    c = set2_(a, (slice(None), 2), -1)
    c.compute(_return_in_memory_array=False)  # don't load into memory
    # za[(slice(None), 2)] = -1  # direct Zarr way (not distributed)

    res = open_storage_array(store, mode="r")

    assert_array_equal(
        res[:],
        np.array([[1, 2, -1, 4], [5, 6, -1, 8], [9, 10, -1, 12], [13, 14, -1, 16]]),
    )
