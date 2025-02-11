import random

import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.backend_array_api import namespace as nxp
from cubed.tests.utils import MAIN_EXECUTORS


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


def test_random(spec, executor):
    a = cubed.random.random((10, 10), chunks=(4, 5), spec=spec)

    assert a.shape == (10, 10)
    assert a.chunks == ((4, 4, 2), (5, 5))
    assert a.dtype == xp.float64

    x = nxp.unique_values(a.compute(executor=executor))
    assert x.dtype == xp.float64
    assert len(x) > 90


def test_random_dtype(spec, executor):
    a = cubed.random.random((10, 10), dtype=xp.float32, chunks=(4, 5), spec=spec)

    assert a.shape == (10, 10)
    assert a.chunks == ((4, 4, 2), (5, 5))
    assert a.dtype == xp.float32

    x = nxp.unique_values(a.compute(executor=executor))
    assert x.dtype == xp.float32
    assert len(x) > 90


def test_random_add(spec, executor):
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)

    c = xp.add(a, b)

    x = nxp.unique_values(c.compute(executor=executor))
    assert len(x) > 90


def test_random_seed(spec, executor):
    random.seed(42)
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    a_result = a.compute(executor=executor)

    random.seed(42)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b_result = b.compute(executor=executor)

    assert_array_equal(a_result, b_result)
