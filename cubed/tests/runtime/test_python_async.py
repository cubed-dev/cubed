import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

from cubed.runtime.executors.python_async import map_unordered
from cubed.tests.runtime.utils import check_invocation_counts, deterministic_failure


async def run_test(function, input, retries=2, use_backups=False, batch_size=None):
    outputs = set()
    concurrent_executor = ThreadPoolExecutor()
    try:
        async for output in map_unordered(
            concurrent_executor,
            function,
            input,
            retries=retries,
            use_backups=use_backups,
            batch_size=batch_size,
        ):
            outputs.add(output)
    finally:
        concurrent_executor.shutdown(wait=False)
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


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        ({0: [60]}, 10, 2),
    ],
)
# fmt: on
@pytest.mark.skip(reason="This passes, but Python will not exit until the slow task is done.")
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


def test_batch(tmp_path):
    # input is unbounded, so if entire input were consumed and not read
    # in batches then it would never return, since it would never
    # run the first (failing) input
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_test(
                function=partial(deterministic_failure, tmp_path, {0: [-1]}),
                input=itertools.count(),
                retries=0,
                batch_size=10,
            )
        )
