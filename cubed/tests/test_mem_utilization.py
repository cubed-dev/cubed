import math
import shutil
from functools import partial, reduce

import pytest

from cubed.core.ops import partial_reduce
from cubed.core.optimization import multiple_inputs_optimize_dag

pytest.importorskip("lithops")

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.backend_array_api import namespace as nxp
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
def test_index(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[1:, :]
    run_operation(tmp_path, "index", b)


@pytest.mark.slow
def test_index_step(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[::2, :]
    run_operation(tmp_path, "index_step", b)


# Creation Functions


@pytest.mark.slow
def test_eye(tmp_path, spec):
    a = xp.eye(10000, 10000, chunks=(5000, 5000), spec=spec)
    run_operation(tmp_path, "eye", a)


@pytest.mark.slow
def test_tril(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.tril(a)
    run_operation(tmp_path, "tril", b)


# Elementwise Functions


@pytest.mark.slow
def test_add(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    run_operation(tmp_path, "add", c)


@pytest.mark.slow
def test_add_reduce_left(tmp_path, spec):
    # Perform the `add` operation repeatedly on pairs of arrays, also known as fold left.
    # See https://en.wikipedia.org/wiki/Fold_(higher-order_function)
    #
    # o   o
    #  \ /
    #   o   o
    #    \ /
    #     o   o
    #      \ /
    #       o
    #
    # Fusing fold left operations will result in a single fused operation.
    n_arrays = 10
    arrs = [
        cubed.random.random((10000, 10000), chunks=(5000, 5000), spec=spec)
        for _ in range(n_arrays)
    ]
    result = reduce(lambda x, y: xp.add(x, y), arrs)
    opt_fn = partial(multiple_inputs_optimize_dag, max_total_source_arrays=n_arrays * 2)
    run_operation(tmp_path, "add_reduce_left", result, optimize_function=opt_fn)


@pytest.mark.slow
def test_add_reduce_right(tmp_path, spec):
    # Perform the `add` operation repeatedly on pairs of arrays, also known as fold right.
    # See https://en.wikipedia.org/wiki/Fold_(higher-order_function)
    #
    #     o   o
    #      \ /
    #   o   o
    #    \ /
    # o   o
    #  \ /
    #   o
    #
    # Note that fusing fold right operations will result in unbounded memory usage unless care
    # is taken to limit fusion - which `multiple_inputs_optimize_dag` will do, with the result
    # that there is more than one fused operation (not a single fused operation).
    n_arrays = 10
    arrs = [
        cubed.random.random((10000, 10000), chunks=(5000, 5000), spec=spec)
        for _ in range(n_arrays)
    ]
    result = reduce(lambda x, y: xp.add(y, x), reversed(arrs))
    opt_fn = partial(multiple_inputs_optimize_dag, max_total_source_arrays=n_arrays * 2)
    run_operation(tmp_path, "add_reduce_right", result, optimize_function=opt_fn)


@pytest.mark.slow
def test_negative(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.negative(a)
    run_operation(tmp_path, "negative", b)


# Linear Algebra Functions


@pytest.mark.slow
def test_matmul(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)
    run_operation(tmp_path, "matmul", e)


@pytest.mark.slow
def test_matrix_transpose(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.matrix_transpose(a)
    run_operation(tmp_path, "matrix_transpose", b)


@pytest.mark.slow
def test_tensordot(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.tensordot(c, d, axes=1)
    run_operation(tmp_path, "tensordot", e)


# Manipulation Functions


@pytest.mark.slow
def test_concat(tmp_path, spec):
    # Note 'a' has one fewer element in axis=0 to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.concat((a, b), axis=0)
    run_operation(tmp_path, "concat", c)


@pytest.mark.slow
def test_reshape(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    # need intermediate reshape due to limitations in Dask's reshape_rechunk
    b = xp.reshape(a, (5000, 2, 10000))
    c = xp.reshape(b, (5000, 20000))
    run_operation(tmp_path, "reshape", c)


@pytest.mark.slow
def test_stack(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.stack((a, b), axis=0)
    run_operation(tmp_path, "stack", c)


# Searching Functions


@pytest.mark.slow
def test_argmax(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.argmax(a, axis=0)
    run_operation(tmp_path, "argmax", b)


# Statistical Functions


@pytest.mark.slow
def test_max(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.max(a, axis=0)
    run_operation(tmp_path, "max", b)


@pytest.mark.slow
def test_mean(tmp_path, spec):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.mean(a, axis=0)
    run_operation(tmp_path, "mean", b)


@pytest.mark.slow
def test_sum_partial_reduce(tmp_path, spec):
    a = cubed.random.random(
        (40000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = partial_reduce(a, nxp.sum, split_every={0: 8})
    run_operation(tmp_path, "sum_partial_reduce", b)


# Internal functions


def run_operation(tmp_path, name, result_array, *, optimize_function=None):
    # result_array.visualize(f"cubed-{name}-unoptimized", optimize_graph=False)
    # result_array.visualize(f"cubed-{name}", optimize_function=optimize_function)
    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
    hist = HistoryCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(
        result_array,
        store=None,
        executor=executor,
        callbacks=[hist],
        optimize_function=optimize_function,
    )

    df = hist.stats_df
    print(df)

    # check projected_mem_utilization does not exceed 1
    assert (df["projected_mem_utilization"] <= 1.0).all()

    # delete temp files for this test immediately since they are so large
    shutil.rmtree(tmp_path)
