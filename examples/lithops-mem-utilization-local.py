import logging

import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.tqdm import TqdmProgressBar
from cubed.runtime.executors.lithops import LithopsDagExecutor

logging.basicConfig(level=logging.INFO)
# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

LITHOPS_LOCAL_CONFIG = {"lithops": {"backend": "localhost", "storage": "localhost"}}

# Array Object


def run_index():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = a[1:, :]
    run_operation("index", b)


# Creation Functions


def run_eye():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = xp.eye(10000, 10000, chunks=(5000, 5000), spec=spec)
    run_operation("eye", a)


def run_tril():
    spec = cubed.Spec(None, max_mem=1_300_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.tril(a)
    run_operation("tril", b)


# Elementwise Functions


def run_add():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    run_operation("add", c)


def run_negative():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.negative(a)
    run_operation("negative", b)


# Linear Algebra Functions


def run_matmul():
    spec = cubed.Spec(None, max_mem=1_000_000_000)
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


def run_matrix_transpose():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.matrix_transpose(a)
    run_operation("matrix_transpose", b)


def run_tensordot():
    spec = cubed.Spec(None, max_mem=1_000_000_000)
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


def run_concat():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    # Note 'a' has one fewer element in axis=0 to force chunking to cross array boundaries
    a = cubed.random.random(
        (9999, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.concat((a, b), axis=0)
    run_operation("concat", c)


def run_reshape():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    # need intermediate reshape due to limitations in Dask's reshape_rechunk
    b = xp.reshape(a, (5000, 2, 10000))
    c = xp.reshape(b, (5000, 20000))
    run_operation("reshape", c)


def run_stack():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.stack((a, b), axis=0)
    run_operation("stack", c)


# Searching Functions


def run_argmax():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.argmax(a, axis=0)
    run_operation("argmax", b)


# Statistical Functions


def run_max():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.max(a, axis=0)
    run_operation("max", b)


def run_mean():
    spec = cubed.Spec(None, max_mem=1_200_000_000)
    a = cubed.random.random(
        (10000, 10000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = xp.mean(a, axis=0)
    run_operation("mean", b)


def run_operation(name, result_array):
    print(name)
    result_array.visualize(f"cubed-{name}-unoptimized", optimize_graph=False)
    result_array.visualize(f"cubed-{name}")
    executor = LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG)
    with logging_redirect_tqdm():
        progress = TqdmProgressBar()
        hist = HistoryCallback()
        # use store=None to write to temporary zarr
        cubed.to_zarr(
            result_array, store=None, executor=executor, callbacks=[progress, hist]
        )

    plan_df = pd.read_csv(hist.plan_df_path)
    stats_df = pd.read_csv(hist.stats_df_path)
    df = analyze(plan_df, stats_df)
    print(df)
    print()


def analyze(plan_df, stats_df):

    # this was found by looking at peak_mem_end_mb for a job with a tiny amount of data
    baseline_mem_mb = 110

    # convert memory to MB
    plan_df["required_mem_mb"] = plan_df["required_mem"] / 1_000_000
    plan_df["total_mem_mb"] = plan_df["required_mem_mb"] + baseline_mem_mb
    plan_df = plan_df[
        ["array_name", "op_name", "required_mem_mb", "total_mem_mb", "num_tasks"]
    ]
    stats_df["peak_mem_start_mb"] = stats_df["peak_memory_start"] / 1_000_000
    stats_df["peak_mem_end_mb"] = stats_df["peak_memory_end"] / 1_000_000
    stats_df["peak_mem_delta_mb"] = (
        stats_df["peak_mem_end_mb"] - stats_df["peak_mem_start_mb"]
    )

    # find per-array stats
    df = stats_df.groupby("array_name", as_index=False).agg(
        {
            "peak_mem_start_mb": ["min", "mean", "max"],
            "peak_mem_end_mb": ["max"],
            "peak_mem_delta_mb": ["min", "mean", "max"],
        }
    )

    # flatten multi-index
    df.columns = ["_".join(a).rstrip("_") for a in df.columns.to_flat_index()]
    df = df.merge(plan_df, on="array_name")

    def utilization(row):
        return row["peak_mem_end_mb_max"] / row["total_mem_mb"]

    df["utilization"] = df.apply(lambda row: utilization(row), axis=1)
    df = df[
        [
            "array_name",
            "op_name",
            "num_tasks",
            "peak_mem_start_mb_max",
            "peak_mem_end_mb_max",
            "peak_mem_delta_mb_max",
            "required_mem_mb",
            "total_mem_mb",
            "utilization",
        ]
    ]

    return df


if __name__ == "__main__":
    run_index()

    run_eye()
    run_tril()

    run_add()
    run_negative()

    run_matmul()
    run_matrix_transpose()
    run_tensordot()

    run_concat()
    run_reshape()
    run_stack()

    run_argmax()

    run_max()
    run_mean()
