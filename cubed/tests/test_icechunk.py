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

from icechunk import Repository, Storage

from cubed.icechunk import store_icechunk


@pytest.fixture(
    scope="module",
    params=MAIN_EXECUTORS,
    ids=[executor.name for executor in MAIN_EXECUTORS],
)
def executor(request):
    return request.param


@pytest.fixture(scope="function")
def icechunk_storage(tmpdir) -> "Storage":
    return Storage.new_local_filesystem(str(tmpdir))


def create_icechunk(a, icechunk_storage, /, *, dtype=None, chunks=None):
    # from dask.asarray
    if not isinstance(getattr(a, "shape", None), Iterable):
        # ensure blocks are arrays
        a = np.asarray(a, dtype=dtype)
    if dtype is None:
        dtype = a.dtype

    repo = Repository.create(storage=icechunk_storage)
    session = repo.writable_session("main")
    store = session.store

    group = zarr.group(store=store, overwrite=True)
    arr = group.create_array("a", shape=a.shape, dtype=dtype, chunks=chunks)

    arr[...] = a

    session.commit("commit 1")


def test_from_zarr_icechunk(icechunk_storage, executor):
    create_icechunk(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        icechunk_storage,
        chunks=(2, 2),
    )

    repo = Repository.open(icechunk_storage)
    session = repo.readonly_session(branch="main")
    store = session.store

    a = cubed.from_zarr(store, path="a")
    assert_array_equal(
        a.compute(executor=executor), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )


def test_store_icechunk(icechunk_storage, executor):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))

    repo = Repository.create(storage=icechunk_storage)
    session = repo.writable_session("main")
    with session.allow_pickling():
        store = session.store
        group = zarr.group(store=store, overwrite=True)
        target = group.create_array(
            "a", shape=a.shape, dtype=a.dtype, chunks=a.chunksize
        )
        store_icechunk(session, sources=a, targets=target, executor=executor)
    session.commit("commit 1")

    # reopen store and check contents of array
    repo = Repository.open(icechunk_storage)
    session = repo.readonly_session(branch="main")
    store = session.store

    group = zarr.open_group(store=store, mode="r")
    assert_array_equal(
        cubed.from_array(group["a"])[:], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
