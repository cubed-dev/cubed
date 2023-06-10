import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

from cubed.runtime.executors.python_async import map_unordered
from cubed.tests.runtime.utils import check_invocation_counts, deterministic_failure


async def run_test(function, input, retries=2):
    outputs = set()
    with ThreadPoolExecutor() as concurrent_executor:
        async for output in map_unordered(
            concurrent_executor, function, input, retries=retries
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
    check_invocation_counts(tmp_path, timing_map, n_tasks)


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
