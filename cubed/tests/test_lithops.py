import pytest
from lithops.executors import FunctionExecutor

from cubed.runtime.executors.lithops import map_with_retries

# Functions read and write from local filesystem since lithops runs tasks in a separate process


def read_int_from_file(path):
    with open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with open(path, "w") as f:
        f.write(str(i))


def test_map_with_retries_no_failures(tmp_path):
    def never_fail(i):
        invocation_count_file = tmp_path / f"{i}"
        write_int_to_file(invocation_count_file, 1)
        return i

    with FunctionExecutor() as executor:
        map_with_retries(executor, never_fail, range(3), max_failures=0)

    assert read_int_from_file(tmp_path / "0") == 1
    assert read_int_from_file(tmp_path / "1") == 1
    assert read_int_from_file(tmp_path / "2") == 1


def test_map_with_retries(tmp_path):
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
        with FunctionExecutor() as executor:
            map_with_retries(
                executor, fail_on_first_invocation, [0, 1, 2], max_failures=2
            )

    with FunctionExecutor() as executor:
        map_with_retries(executor, fail_on_first_invocation, [3, 4, 5], max_failures=3)

    assert read_int_from_file(tmp_path / "3") == 2
    assert read_int_from_file(tmp_path / "4") == 2
    assert read_int_from_file(tmp_path / "5") == 2
