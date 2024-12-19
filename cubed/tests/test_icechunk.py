from typing import Iterable

import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.tests.utils import MAIN_EXECUTORS

icechunk = pytest.importorskip("icechunk")

from cubed.icechunk import store_icechunk


@pytest.fixture(
    scope="module",
    params=MAIN_EXECUTORS,
    ids=[executor.name for executor in MAIN_EXECUTORS],
)
def executor(request):
    return request.param


def create_icechunk(a, tmp_path, /, *, dtype=None, chunks=None):
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    store = icechunk.IcechunkStore.create(
        storage=icechunk.StorageConfig.filesystem(tmp_path / "icechunk"),
        config=icechunk.StoreConfig(inline_chunk_threshold_bytes=1),
        read_only=False,
    )

    group = zarr.group(store=store, overwrite=True)
    arr = group.create_array("a", shape=a.shape, chunk_shape=chunks, dtype=dtype)

    arr[...] = a

    store.commit("commit 1")


def test_from_zarr_icechunk(tmp_path, executor):
    create_icechunk(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        tmp_path,
        chunks=(2, 2),
    )

    store = icechunk.IcechunkStore.open_existing(
        storage=icechunk.StorageConfig.filesystem(tmp_path / "icechunk"),
    )

    a = cubed.from_zarr(store, path="a")
    assert_array_equal(
        a.compute(executor=executor), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )


def test_store_icechunk(tmp_path, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))

    store = icechunk.IcechunkStore.create(
        storage=icechunk.StorageConfig.filesystem(tmp_path / "icechunk"),
        config=icechunk.StoreConfig(inline_chunk_threshold_bytes=1),
        read_only=False,
    )
    with store.preserve_read_only():
        group = zarr.group(store=store, overwrite=True)
        target = group.create_array(
            "a", shape=a.shape, chunk_shape=a.chunksize, dtype=a.dtype
        )
        store_icechunk(store, sources=a, targets=target, executor=executor)
        store.commit("commit 1")

    # reopen store and check contents of array
    store = icechunk.IcechunkStore.open_existing(
        storage=icechunk.StorageConfig.filesystem(tmp_path / "icechunk"),
    )
    group = zarr.open_group(store=store, mode="r")
    assert_array_equal(
        cubed.from_array(group["a"])[:], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
