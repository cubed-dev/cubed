import math
import platform
import shutil
import sys
from functools import partial, reduce

import pandas as pd
import pytest

pytest.importorskip("memray")

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import partial_reduce
from cubed.core.optimization import multiple_inputs_optimize_dag
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.mem_warn import MemoryWarningCallback
from cubed.diagnostics.memray import MemrayCallback
from cubed.runtime.create import create_executor
from cubed.tests.test_core import sqrts
from cubed.tests.utils import LITHOPS_LOCAL_CONFIG

pd.set_option("display.max_columns", None)


ALLOWED_MEM = 2_000_000_000

EXECUTORS = {}

if platform.system() != "Windows":
    EXECUTORS["processes"] = create_executor("processes")

    # Run with max_tasks_per_child=1 so that each task is run in a new process,
    # allowing us to perform a stronger check on peak memory
    if sys.version_info >= (3, 11):
        executor_options = dict(max_tasks_per_child=1)
        EXECUTORS["processes-single-task"] = create_executor(
            "processes", executor_options
        )

try:
    executor_options = dict(config=LITHOPS_LOCAL_CONFIG, wait_dur_sec=0.1)
    EXECUTORS["lithops"] = create_executor("lithops", executor_options)
except ImportError:
    pass


@pytest.fixture()
def spec(tmp_path, reserved_mem):
    return cubed.Spec(tmp_path, allowed_mem=ALLOWED_MEM, reserved_mem=reserved_mem)


@pytest.fixture(
    scope="module",
    params=EXECUTORS.values(),
    ids=EXECUTORS.keys(),
)
def executor(request):
    return request.param


@pytest.fixture(scope="module")
def reserved_mem(executor):
    res = cubed.measure_reserved_mem(executor) * 1.1  # add some wiggle room
    return round_up_to_multiple(res, 10_000_000)  # round up to nearest multiple of 10MB


def round_up_to_multiple(x, multiple=10):
    """Round up to the nearest multiple"""
    return math.ceil(x / multiple) * multiple


# Array Object


@pytest.mark.slow
def test_index(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[1:, :]
    run_operation(tmp_path, executor, "index", b)


@pytest.mark.slow
def test_index_chunk_aligned(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[0:5000, :]
    run_operation(tmp_path, executor, "index_chunk_aligned", b)


@pytest.mark.slow
def test_index_multiple_axes(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[1:, 1:]
    run_operation(tmp_path, executor, "index_multiple_axes", b)


@pytest.mark.slow
def test_index_step(tmp_path, spec, executor):
    # use 400MB chunks so that intermediate after indexing has 200MB chunks
    a = cubed.random.random(
        (20000, 10000), chunks=(10000, 5000), spec=spec
    )  # 400MB chunks
    b = a[::2, :]
    run_operation(tmp_path, executor, "index_step", b)


# Creation Functions


@pytest.mark.slow
def test_eye(tmp_path, spec, executor):
    a = xp.eye(10000, 10000, chunks=(5000, 5000), spec=spec)
    run_operation(tmp_path, executor, "eye", a)


@pytest.mark.slow
def test_tril(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.tril(a)
    run_operation(tmp_path, executor, "tril", b)


# Elementwise Functions


@pytest.mark.slow
@pytest.mark.parametrize("optimize_graph", [False, True])
def test_add(tmp_path, spec, executor, optimize_graph):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    run_operation(tmp_path, executor, "add", c, optimize_graph=optimize_graph)


@pytest.mark.slow
def test_add_reduce_left(tmp_path, spec, executor):
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
    run_operation(
        tmp_path, executor, "add_reduce_left", result, optimize_function=opt_fn
    )


@pytest.mark.slow
def test_add_reduce_right(tmp_path, spec, executor):
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
    run_operation(
        tmp_path, executor, "add_reduce_right", result, optimize_function=opt_fn
    )


@pytest.mark.slow
def test_negative(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.negative(a)
    run_operation(tmp_path, executor, "negative", b)


# Linear Algebra Functions


@pytest.mark.slow
def test_matmul(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)
    run_operation(tmp_path, executor, "matmul", e)


@pytest.mark.slow
def test_matrix_transpose(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.matrix_transpose(a)
    run_operation(tmp_path, executor, "matrix_transpose", b)


@pytest.mark.slow
def test_tensordot(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.tensordot(c, d, axes=1)
    run_operation(tmp_path, executor, "tensordot", e)


# Manipulation Functions


@pytest.mark.slow
def test_concat(tmp_path, spec, executor):
    # Note 'a' has one fewer element in axis=0 to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.concat((a, b), axis=0)
    run_operation(tmp_path, executor, "concat", c)


@pytest.mark.slow
def test_flip(tmp_path, spec, executor):
    # Note 'a' has one fewer element in axis=0 to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.flip(a, axis=0)
    run_operation(tmp_path, executor, "flip", b)


@pytest.mark.slow
def test_flip_multiple_axes(tmp_path, spec, executor):
    # Note 'a' has one fewer element in both axes to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 9999), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.flip(a)
    run_operation(tmp_path, executor, "flip_multiple_axes", b)


@pytest.mark.slow
def test_repeat(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.repeat(a, 3, axis=0)
    run_operation(tmp_path, executor, "repeat", b)


@pytest.mark.slow
def test_reshape(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    # need intermediate reshape due to limitations in Dask's reshape_rechunk
    b = xp.reshape(a, (5000, 2, 10000))
    c = xp.reshape(b, (5000, 20000))
    run_operation(tmp_path, executor, "reshape", c)


@pytest.mark.slow
def test_stack(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.stack((a, b), axis=0)
    run_operation(tmp_path, executor, "stack", c)


@pytest.mark.slow
def test_unstack(tmp_path, spec, executor):
    a = cubed.random.random(
        (2, 10000, 10000), chunks=(2, 5000, 5000), spec=spec
    )  # 400MB chunks
    b, c = xp.unstack(a)
    run_operation(tmp_path, executor, "unstack", b, c)


# Searching Functions


@pytest.mark.slow
def test_argmax(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.argmax(a, axis=0)
    run_operation(tmp_path, executor, "argmax", b)


# Statistical Functions


@pytest.mark.slow
def test_max(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.max(a, axis=0)
    run_operation(tmp_path, executor, "max", b)


@pytest.mark.slow
def test_mean(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.mean(a, axis=0)
    run_operation(tmp_path, executor, "mean", b)


@pytest.mark.slow
def test_sum_partial_reduce(tmp_path, spec, executor):
    a = cubed.random.random(
        (40000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = partial_reduce(a, nxp.sum, split_every={0: 8})
    run_operation(tmp_path, executor, "sum_partial_reduce", b)


# Linear algebra extension


@pytest.mark.slow
def test_qr(tmp_path, spec, executor):
    a = cubed.random.random(
        (40000, 1000), chunks=(5000, 1000), spec=spec
    )  # 40MB chunks
    q, r = xp.linalg.qr(a)
    # don't optimize graph so we use as much memory as possible (reading from Zarr)
    run_operation(tmp_path, executor, "qr", q, r, optimize_graph=False)


# Multiple outputs


@pytest.mark.slow
def test_sqrts(tmp_path, spec, executor):
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b, c = sqrts(a)
    # don't optimize graph so we use as much memory as possible (reading from Zarr)
    run_operation(tmp_path, executor, "sqrts", b, c, optimize_graph=False)


# Internal functions


def run_operation(
    tmp_path,
    executor,
    name,
    *results,
    optimize_graph=True,
    optimize_function=None,
):
    # cubed.visualize(
    #     *results, filename=f"cubed-{name}-unoptimized", optimize_graph=False, show_hidden=True
    # )
    # cubed.visualize(
    #     *results, filename=f"cubed-{name}", optimize_function=optimize_function
    # )
    hist = HistoryCallback()
    mem_warn = MemoryWarningCallback()
    memray = MemrayCallback(mem_threshold=30_000_000)
    # use None for each store to write to temporary zarr
    cubed.store(
        results,
        (None,) * len(results),
        executor=executor,
        callbacks=[hist, mem_warn, memray],
        optimize_graph=optimize_graph,
        optimize_function=optimize_function,
    )

    df = hist.stats_df
    print(df)

    # check peak memory does not exceed allowed mem
    assert (df["peak_measured_mem_end_mb_max"] <= ALLOWED_MEM // 1_000_000).all()

    # check change in peak memory is no more than projected mem
    assert (df["peak_measured_mem_delta_mb_max"] <= df["projected_mem_mb"]).all()

    # check memray peak memory allocated is no more than projected mem
    for op_name, stats in memray.stats.items():
        assert (
            stats.peak_memory_allocated
            <= df.query(f"name=='{op_name}'")["projected_mem_mb"].item() * 1_000_000
        ), f"projected mem exceeds memray's peak allocated for {op_name}"

    # check projected_mem_utilization does not exceed 1
    # except on processes executor that runs multiple tasks in a process
    if (
        executor.name != "processes"
        or executor.kwargs.get("max_tasks_per_child", None) == 1
    ):
        assert (df["projected_mem_utilization"] <= 1.0).all()

    # delete temp files for this test immediately since they are so large
    shutil.rmtree(tmp_path)
