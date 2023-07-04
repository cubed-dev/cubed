import math

import pytest

pytest.importorskip("lithops")

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.runtime.executors.lithops import LithopsDagExecutor
from cubed.tests.utils import LITHOPS_LOCAL_CONFIG


@pytest.fixture()
def spec(tmp_path, reserved_mem):
    return cubed.Spec(tmp_path, allowed_mem=2_000_000_000, reserved_mem=reserved_mem)


@pytest.fixture(scope="module")
def reserved_mem():
    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
    res = cubed.measure_reserved_mem(executor) * 1.1  # add some wiggle room
    return round_up_to_multiple(res, 10_000_000)  # round up to nearest multiple of 10MB


def round_up_to_multiple(x, multiple=10):
    """Round up to the nearest multiple"""
    return math.ceil(x / multiple) * multiple


# Array Object


@pytest.mark.slow
def test_index(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[1:, :]
    run_operation("index", b)


# Creation Functions


@pytest.mark.slow
def test_eye(spec):
    a = xp.eye(10000, 10000, chunks=(5000, 5000), spec=spec)
    run_operation("eye", a)


@pytest.mark.slow
def test_tril(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.tril(a)
    run_operation("tril", b)


# Elementwise Functions


@pytest.mark.slow
def test_add(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    run_operation("add", c)


@pytest.mark.slow
def test_negative(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.negative(a)
    run_operation("negative", b)


# Linear Algebra Functions


@pytest.mark.slow
def test_matmul(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)
    run_operation("matmul", e)


@pytest.mark.slow
def test_matrix_transpose(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.matrix_transpose(a)
    run_operation("matrix_transpose", b)


@pytest.mark.slow
def test_tensordot(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.tensordot(c, d, axes=1)
    run_operation("tensordot", e)


# Manipulation Functions


@pytest.mark.slow
def test_concat(spec):
    # Note 'a' has one fewer element in axis=0 to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.concat((a, b), axis=0)
    run_operation("concat", c)


@pytest.mark.slow
def test_reshape(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    # need intermediate reshape due to limitations in Dask's reshape_rechunk
    b = xp.reshape(a, (5000, 2, 10000))
    c = xp.reshape(b, (5000, 20000))
    run_operation("reshape", c)


@pytest.mark.slow
def test_stack(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.stack((a, b), axis=0)
    run_operation("stack", c)


# Searching Functions


@pytest.mark.slow
def test_argmax(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.argmax(a, axis=0)
    run_operation("argmax", b)


# Statistical Functions


@pytest.mark.slow
def test_max(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.max(a, axis=0)
    run_operation("max", b)


@pytest.mark.slow
def test_mean(spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.mean(a, axis=0)
    run_operation("mean", b)


# Internal functions


def run_operation(name, result_array):
    # result_array.visualize(f"cubed-{name}-unoptimized", optimize_graph=False)
    # result_array.visualize(f"cubed-{name}")
    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
    hist = HistoryCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(result_array, store=None, executor=executor, callbacks=[hist])

    df = hist.stats_df
    print(df)

    # check projected_mem_utilization does not exceed 1
    assert (df["projected_mem_utilization"] <= 1.0).all()
