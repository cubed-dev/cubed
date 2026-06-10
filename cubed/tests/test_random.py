import math
import random

import numpy as np
import pytest

import cubed
import cubed.array_api as xp
import cubed.random
from cubed._testing import assert_array_equal
from cubed.backend_array_api import namespace as nxp
from cubed.random import _wang_hash
from cubed.tests.utils import MAIN_EXECUTORS


@pytest.fixture
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


@pytest.fixture(
    scope="module",
    params=MAIN_EXECUTORS,
    ids=[executor.name for executor in MAIN_EXECUTORS],
)
def executor(request):
    return request.param


def test_random(spec, executor):
    a = cubed.random.random((10, 10), chunks=(4, 5), spec=spec)

    assert a.shape == (10, 10)
    assert a.chunks == ((4, 4, 2), (5, 5))
    assert a.dtype == xp.float64

    x = nxp.unique_values(a.compute(executor=executor))
    assert x.dtype == xp.float64
    assert x.shape[0] > 90


def test_random_dtype(spec, executor):
    a = cubed.random.random((10, 10), dtype=xp.float32, chunks=(4, 5), spec=spec)

    assert a.shape == (10, 10)
    assert a.chunks == ((4, 4, 2), (5, 5))
    assert a.dtype == xp.float32

    x = nxp.unique_values(a.compute(executor=executor))
    assert x.dtype == xp.float32
    assert x.shape[0] > 90


def test_random_add(spec, executor):
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)

    c = xp.add(a, b)

    x = nxp.unique_values(c.compute(executor=executor))
    assert x.shape[0] > 90


def test_random_seed(spec, executor):
    random.seed(42)
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    a_result = a.compute(executor=executor)

    random.seed(42)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b_result = b.compute(executor=executor)

    assert_array_equal(a_result, b_result)


def test_integers():
    a = cubed.random.integers((10, 10), chunks=(4, 5))

    assert a.shape == (10, 10)
    assert a.chunks == ((4, 4, 2), (5, 5))
    assert a.dtype == nxp.int32

    result = a.compute()

    # values match wang_hash(flat_index)
    shape = (10, 10)
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    for idx in [(0, 0), (0, 1), (3, 4), (9, 9)]:
        flat = sum(i * s for i, s in zip(idx, strides))
        assert result[idx] == _wang_hash(np.array([flat], dtype=np.uint32))[0]

    # no constant-delta pattern (output is incompressible)
    deltas = nxp.diff(nxp.astype(nxp.reshape(result, (-1,)), nxp.int64))
    assert not nxp.all(deltas == deltas[0])


def test_integers_dtype():
    a = cubed.random.integers((10, 10), dtype=nxp.int64, chunks=(4, 5))

    assert a.dtype == nxp.int64
    result = a.compute()
    assert result.dtype == nxp.int64


def test_integers_chunking_independent():
    # same logical array, different chunking — values must be identical
    shape = (10, 8)
    a = cubed.random.integers(shape, chunks=(5, 4))
    b = cubed.random.integers(shape, chunks=(2, 8))

    assert_array_equal(a.compute(), b.compute())
