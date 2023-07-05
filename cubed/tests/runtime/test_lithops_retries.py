import pytest

pytest.importorskip("lithops")

from lithops.executors import LocalhostExecutor

from cubed.runtime.executors.lithops_retries import map_with_retries, wait_with_retries
from cubed.tests.runtime.utils import check_invocation_counts, deterministic_failure


def run_test(function, input, retries, timeout=10):
    with LocalhostExecutor() as executor:
        futures = map_with_retries(
            executor,
            function,
            input,
            timeout=timeout,
            retries=retries,
        )
        done, pending = wait_with_retries(executor, futures, throw_except=False)
        assert len(pending) == 0
    outputs = set(f.result() for f in done)
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
        # first input sleeps once
        ({0: [20]}, 3, 2),
    ],
)
# fmt: on
def test_success(tmp_path, timing_map, n_tasks, retries):
    partial_map_function = lambda x: deterministic_failure(tmp_path, timing_map, x)
    outputs = run_test(
        function=partial_map_function,
        input=range(n_tasks),
        retries=retries,
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
    partial_map_function = lambda x: deterministic_failure(tmp_path, timing_map, x)
    with pytest.raises(RuntimeError):
        run_test(
            function=partial_map_function,
            input=range(n_tasks),
            retries=retries,
        )

    check_invocation_counts(tmp_path, timing_map, n_tasks, retries)
