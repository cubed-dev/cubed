import random

import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.tests.utils import MAIN_EXECUTORS


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


@pytest.fixture(scope="module", params=MAIN_EXECUTORS)
def executor(request):
    return request.param


def test_random(spec, executor):
    a = cubed.random.random((10, 10), chunks=(4, 5), spec=spec)

    assert a.shape == (10, 10)
    assert a.chunks == ((4, 4, 2), (5, 5))

    x = set(a.compute(executor=executor).flat)
    assert len(x) > 90


def test_random_add(spec, executor):
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)

    c = xp.add(a, b)

    x = set(c.compute(executor=executor).flat)
    assert len(x) > 90


def test_random_seed(spec, executor):
    random.seed(42)
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    a_result = a.compute(executor=executor)

    random.seed(42)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b_result = b.compute(executor=executor)

    assert_array_equal(a_result, b_result)
