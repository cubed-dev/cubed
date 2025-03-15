import math
import platform
import sys

import pytest

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.runtime.create import create_executor
from cubed.tests.test_mem_utilization import run_operation
from cubed.tests.utils import LITHOPS_LOCAL_CONFIG

ALLOWED_MEM = 2_500_000_000

EXECUTORS = {}

if platform.system() != "Windows":
    # Run with max_tasks_per_child=1 so that each task is run in a new process,
    # allowing us to perform a stronger check on peak memory
    if sys.version_info >= (3, 11):
        executor_options = dict(max_tasks_per_child=1, max_workers=4)
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


@pytest.mark.slow
def test_rechunk_era5(tmp_path, spec, executor):
    # This example is based on rechunking an ERA5 dataset
    # from https://github.com/pangeo-data/rechunker/pull/89
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    x = cubed.random.random(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    i = 0
    for copy_chunks, target_chunks in _rechunk_plan(x, target_chunks):
        # Find the smallest shape that contains the three chunk sizes
        # This will be a lot less than the full ERA5 shape (350640, 721, 1440),
        # making it suitable for running in a test
        test_shape = tuple(
            max(a, b, c) for a, b, c in zip(source_chunks, copy_chunks, target_chunks)
        )
        print(i, test_shape, source_chunks, copy_chunks, target_chunks)

        a = cubed.random.random(
            test_shape, dtype=xp.float32, chunks=source_chunks, spec=spec
        )
        b = a.rechunk(target_chunks, use_new_impl=True)

        run_operation(tmp_path, executor, f"rechunk_era5_stage_{i}", b)

        source_chunks = target_chunks
        i += 1
