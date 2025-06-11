import re
import time
from urllib.parse import urlparse

import fsspec

from cubed.utils import join_path


def read_int_from_file(path):
    with fsspec.open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with fsspec.open(path, "w") as f:
        f.write(str(i))


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
    invocation_count_file = join_path(path, f"{i}")
    fs = fsspec.open(invocation_count_file).fs
    if fs.exists(invocation_count_file):
        invocation_count = read_int_from_file(invocation_count_file)
    else:
        invocation_count = 0
    write_int_to_file(invocation_count_file, invocation_count + 1)

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
    # see https://filesystem-spec.readthedocs.io/en/latest/async.html#synchronous-api
    if re.match(r"^[a-zA-Z]:\\", str(path)):  # Windows local file
        protocol = ""
    else:
        protocol = urlparse(str(path)).scheme
    fs = fsspec.filesystem(protocol)
    paths = [join_path(path, str(i)) for i in range(n_tasks)]
    out = fs.cat(paths)
    path_to_i = lambda p: int(p.rsplit("/", 1)[-1])
    actual_invocation_counts = {path_to_i(path): int(val) for path, val in out.items()}

    if actual_invocation_counts != expected_invocation_counts:
        for i, expected_count in expected_invocation_counts.items():
            actual_count = actual_invocation_counts[i]
            if actual_count != expected_count:
                print(
                    f"Invocation count for {i}, expected: {expected_count}, actual: {actual_count}"
                )
    assert actual_invocation_counts == expected_invocation_counts
