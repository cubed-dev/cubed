import asyncio
import time
from pathlib import Path

import obstore as obs


def path_to_store(path):
    if isinstance(path, str):
        if "://" not in path:
            return obs.store.from_url(Path(path).as_uri(), mkdir=True)
        else:
            return obs.store.from_url(path)
    elif isinstance(path, Path):
        return obs.store.from_url(path.as_uri(), mkdir=True)


def read_int_from_file(store, path):
    result = obs.get(store, path)
    return int(result.bytes())


def write_int_to_file(store, path, i):
    obs.put(store, path, bytes(str(i), encoding="UTF8"))


def deterministic_failure(path, timing_map, i, *, default_sleep=0.01, name=None):
    """A function that can either run normally, run slowly, or raise
    an exception, depending on input and invocation count.

    The timing_map is a dictionary whose keys are inputs and values
    are sequences of timing information for each invocation.
    The maginitude of the value is the time to sleep in seconds, and
    the sign indicates the input is returned normally (positive, or 0),
    or an exception is raised (negative).

    If a input is missing then all invocations will run normally, with a
    small default sleep to avoid spurious backups being launched.

    If there are subsequent invocations to the ones in the sequence, then
    they will all run normally.
    """
    # increment number of invocations of this function with arg i
    store = path_to_store(path)
    try:
        invocation_count = read_int_from_file(store, f"{i}")
    except FileNotFoundError:
        invocation_count = 0
    write_int_to_file(store, f"{i}", invocation_count + 1)

    timing_code = default_sleep
    if i in timing_map:
        timing_codes = timing_map[i]
        if invocation_count >= len(timing_codes):
            timing_code = default_sleep
        else:
            timing_code = timing_codes[invocation_count]

    if timing_code >= 0:
        time.sleep(timing_code)
        return i
    else:
        time.sleep(-timing_code)
        raise RuntimeError(
            f"Deliberately fail on invocation number {invocation_count + 1} for input {i}"
        )


def check_invocation_counts(
    path, timing_map, n_tasks, retries=None, expected_invocation_counts_overrides=None
):
    asyncio.run(
        check_invocation_counts_async(
            path,
            timing_map,
            n_tasks,
            retries=retries,
            expected_invocation_counts_overrides=expected_invocation_counts_overrides,
        )
    )


async def check_invocation_counts_async(
    path, timing_map, n_tasks, retries=None, expected_invocation_counts_overrides=None
):
    expected_invocation_counts = {}
    for i in range(n_tasks):
        if i not in timing_map:
            expected_invocation_counts[i] = 1
        else:
            timing_codes = timing_map[i]
            expected_invocation_count = len(timing_codes) + 1

            if retries is not None:
                # there shouldn't have been more than retries + 1 invocations
                max_invocations = retries + 1
                expected_invocation_count = min(
                    expected_invocation_count, max_invocations
                )

            expected_invocation_counts[i] = expected_invocation_count

    if expected_invocation_counts_overrides is not None:
        expected_invocation_counts.update(expected_invocation_counts_overrides)

    # retrieve outputs concurrently, so we can test on large numbers of inputs
    store = path_to_store(path)
    paths = [str(i) for i in range(n_tasks)]
    results = await asyncio.gather(*[obs.get_async(store, path) for path in paths])
    values = await asyncio.gather(*[result.bytes_async() for result in results])
    actual_invocation_counts = {i: int(val) for i, val in enumerate(values)}

    if actual_invocation_counts != expected_invocation_counts:
        for i, expected_count in expected_invocation_counts.items():
            actual_count = actual_invocation_counts[i]
            if actual_count != expected_count:
                print(
                    f"Invocation count for {i}, expected: {expected_count}, actual: {actual_count}"
                )
    assert actual_invocation_counts == expected_invocation_counts
