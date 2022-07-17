import time

import pytest

pytest.importorskip("lithops")

from lithops.executors import LocalhostExecutor

from cubed.runtime.executors.lithops import map_unordered

# Functions read and write from local filesystem since lithops runs tasks in a separate process


def read_int_from_file(path):
    with open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with open(path, "w") as f:
        f.write(str(i))


def test_map_unordered_no_failures(tmp_path):
    def never_fail(i):
        invocation_count_file = tmp_path / f"{i}"
        write_int_to_file(invocation_count_file, 1)
        return i

    with LocalhostExecutor() as executor:
        for _ in map_unordered(executor, never_fail, range(3), max_failures=0):
            pass

    assert read_int_from_file(tmp_path / "0") == 1
    assert read_int_from_file(tmp_path / "1") == 1
    assert read_int_from_file(tmp_path / "2") == 1


def test_map_unordered(tmp_path):
    def fail_on_first_invocation(i):
        invocation_count_file = tmp_path / f"{i}"
        if invocation_count_file.exists():
            count = read_int_from_file(invocation_count_file)
            write_int_to_file(invocation_count_file, count + 1)
        else:
            write_int_to_file(invocation_count_file, 1)
            raise RuntimeError(f"Fail on {i}")
        return i

    with pytest.raises(RuntimeError):
        with LocalhostExecutor() as executor:
            for _ in map_unordered(
                executor, fail_on_first_invocation, [0, 1, 2], max_failures=2
            ):
                pass

    with LocalhostExecutor() as executor:
        for _ in map_unordered(
            executor, fail_on_first_invocation, [3, 4, 5], max_failures=3
        ):
            pass

    assert read_int_from_file(tmp_path / "3") == 2
    assert read_int_from_file(tmp_path / "4") == 2
    assert read_int_from_file(tmp_path / "5") == 2


def test_map_unordered_execution_timeout(tmp_path):
    def sleep_on_first_invocation(i):
        invocation_count_file = tmp_path / f"{i}"
        if invocation_count_file.exists():
            count = read_int_from_file(invocation_count_file)
            write_int_to_file(invocation_count_file, count + 1)
        else:
            write_int_to_file(invocation_count_file, 1)
            # only sleep on first invocation of input = 0
            if i == 0:
                time.sleep(60)
        return i

    # set execution timeout to less than sleep value above, to check that
    # task is retried after timeout exception
    config = {"lithops": {"execution_timeout": 30}}
    with LocalhostExecutor(config=config) as executor:
        for _ in map_unordered(executor, sleep_on_first_invocation, [0, 1, 2]):
            pass

    assert read_int_from_file(tmp_path / "0") == 2
    assert read_int_from_file(tmp_path / "1") == 1
    assert read_int_from_file(tmp_path / "2") == 1


def test_map_unordered_stragglers(tmp_path):
    def sleep_on_first_invocation(i):
        invocation_count_file = tmp_path / f"{i}"
        if invocation_count_file.exists():
            count = read_int_from_file(invocation_count_file)
            write_int_to_file(invocation_count_file, count + 1)
        else:
            write_int_to_file(invocation_count_file, 1)
            # only sleep on first invocation of input = 0
            if i == 0:
                time.sleep(30)
        return i

    # TODO: run a test like this using a cloud executor
    # Reason: this test passes, but lithops local mode only runs one job at a time,
    # so it actually waits for the first job to finish before running the second one.

    config = {"lithops": {"log_level": "DEBUG"}}
    with LocalhostExecutor(config=config) as executor:
        for _ in map_unordered(
            executor, sleep_on_first_invocation, range(10), use_backups=True
        ):
            pass

    assert read_int_from_file(tmp_path / "0") == 2
    for i in range(1, 10):
        assert read_int_from_file(tmp_path / f"{i}") == 1
