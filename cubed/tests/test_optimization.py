import random

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed.tests.utils import TaskCounter


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


def test_fusion(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = xp.negative(c)

    assert d.plan.num_tasks(optimize_graph=False) == 12
    assert d.plan.num_tasks(optimize_graph=True) == 4

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == 4

    assert_array_equal(
        result,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32),
    )


def test_fusion_transpose(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = c.T

    assert d.plan.num_tasks(optimize_graph=False) == 12
    assert d.plan.num_tasks(optimize_graph=True) == 4

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == 4

    assert_array_equal(
        result,
        np.array([[-1, -4, -7], [-2, -5, -8], [-3, -6, -9]]).astype(np.float32),
    )


def test_no_fusion(spec):
    # b can't be fused with c because d also depends on b
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.positive(b)
    d = xp.equal(b, c)

    assert d.plan.num_tasks(optimize_graph=False) == 3
    assert d.plan.num_tasks(optimize_graph=True) == 3

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == 3

    assert_array_equal(result, np.ones((2, 2)))


def test_no_fusion_multiple_edges(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.asarray(b)
    # b and c are the same array, so d has a single dependency
    # with multiple edges
    # this should not be fused under the current logic
    d = xp.equal(b, c)

    assert d.plan.num_tasks(optimize_graph=False) == 2
    assert d.plan.num_tasks(optimize_graph=True) == 2

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == 2

    assert_array_equal(result, np.full((2, 2), True))


def test_fusion_multiple_inputs(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
    c = xp.negative(a)
    d = xp.negative(b)
    e = c + d

    e.visualize("cubed-fusion-multiple-inputs-unoptimized", optimize_graph=False)
    e.visualize("cubed-fusion-multiple-inputs", optimize_graph=True)

    assert e.plan.num_tasks(optimize_graph=False) == 12
    assert e.plan.num_tasks(optimize_graph=True) == 4  # TODO: this fails

    task_counter = TaskCounter()
    result = e.compute(callbacks=[task_counter])
    assert task_counter.value == 12

    assert_array_equal(
        result,
        np.array([[-2, -3, -4], [-5, -6, -7], [-8, -9, -10]]),
    )


def test_blockwise_subgraph_fusion(spec):
    # test that a large subgraph of blockwise operations is fused

    random.seed(42)

    a = cubed.random.random((100, 90, 80), chunks=10, spec=spec)
    b = cubed.random.random((100, 90, 80), chunks=10, spec=spec)
    x = cubed.random.random((90, 80), chunks=10, spec=spec)
    y = cubed.random.random((90, 80), chunks=10, spec=spec)

    result = a[1:] * x + b[1:] * y
    result = xp.mean(result)

    result.visualize("cubed-geo-unoptimized", optimize_graph=False)
    result.visualize("cubed-geo", optimize_graph=True)

    # TODO: add assert for num_tasks for optimize_graph=False/True

    assert_array_equal(
        result.compute(optimize_graph=False), result.compute(optimize_graph=True)
    )
