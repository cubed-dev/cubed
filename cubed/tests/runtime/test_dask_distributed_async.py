import asyncio
from functools import partial

import pytest

from cubed.tests.runtime.utils import check_invocation_counts, deterministic_failure

pytest.importorskip("dask.distributed")

from dask.distributed import Client

from cubed.runtime.executors.dask_distributed_async import map_unordered


async def run_test(function, input, retries, use_backups=False):
    outputs = set()
    async with Client(asynchronous=True) as client:
        async for output in map_unordered(
            client,
            function,
            input,
            retries=retries,
            use_backups=use_backups,
        ):
            outputs.add(output)
    return outputs


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        # no failures
        ({}, 3, 2),
        # first invocation fails
        ({0: [-1], 1: [-1], 2: [-1]}, 3, 2),
        # first two invocations fail
        ({0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}, 3, 2),
        # first input sleeps once (not tested since timeout is not supported)
        # ({0: [20]}, 3, 2),
    ],
)
# fmt: on
def test_success(tmp_path, timing_map, n_tasks, retries):
    outputs = asyncio.run(
        run_test(
            function=partial(deterministic_failure, tmp_path, timing_map),
            input=range(n_tasks),
            retries=retries,
        )
    )

    assert outputs == set(range(n_tasks))

    check_invocation_counts(tmp_path, timing_map, n_tasks, retries)


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        # too many failures
        ({0: [-1], 1: [-1], 2: [-1, -1, -1]}, 3, 2),
    ],
)
# fmt: on
def test_failure(tmp_path, timing_map, n_tasks, retries):
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_test(
                function=partial(deterministic_failure, tmp_path, timing_map),
                input=range(n_tasks),
                retries=retries,
            )
        )

    check_invocation_counts(tmp_path, timing_map, n_tasks, retries)


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        ({0: [60]}, 10, 2),
    ],
)
# fmt: on
def test_stragglers(tmp_path, timing_map, n_tasks, retries):
    outputs = asyncio.run(
        run_test(
            function=partial(deterministic_failure, tmp_path, timing_map),
            input=range(n_tasks),
            retries=retries,
            use_backups=True,
        )
    )

    assert outputs == set(range(n_tasks))

    check_invocation_counts(tmp_path, timing_map, n_tasks, retries)
