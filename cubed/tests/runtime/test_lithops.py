from functools import partial

import pytest

from cubed.tests.runtime.utils import (
    fail_on_first_invocation,
    never_fail,
    read_int_from_file,
    sleep_on_first_invocation,
)
from cubed.utils import join_path

pytest.importorskip("lithops")

from lithops.executors import LocalhostExecutor

from cubed.runtime.executors.lithops import map_unordered


def test_map_unordered_no_failures(tmp_path):
    with LocalhostExecutor() as executor:
        for _ in map_unordered(
            executor, partial(never_fail, tmp_path), range(3), max_failures=0
        ):
            pass

    assert read_int_from_file(join_path(tmp_path, "0")) == 1
    assert read_int_from_file(join_path(tmp_path, "1")) == 1
    assert read_int_from_file(join_path(tmp_path, "2")) == 1


def test_map_unordered_recovers_from_failures(tmp_path):
    with LocalhostExecutor() as executor:
        for _ in map_unordered(
            executor,
            partial(fail_on_first_invocation, tmp_path),
            range(3),
            max_failures=3,
        ):
            pass

    assert read_int_from_file(join_path(tmp_path, "0")) == 2
    assert read_int_from_file(join_path(tmp_path, "1")) == 2
    assert read_int_from_file(join_path(tmp_path, "2")) == 2


def test_map_unordered_too_many_failures(tmp_path):
    with pytest.raises(RuntimeError):
        with LocalhostExecutor() as executor:
            for _ in map_unordered(
                executor,
                partial(fail_on_first_invocation, tmp_path),
                range(3),
                max_failures=2,
            ):
                pass


def test_map_unordered_execution_timeout(tmp_path):
    # set execution timeout to less than sleep value in sleep_on_first_invocation,
    # to check that task is retried after timeout exception
    config = {"lithops": {"execution_timeout": 30}}
    with LocalhostExecutor(config=config) as executor:
        for _ in map_unordered(
            executor, partial(sleep_on_first_invocation, tmp_path), range(3)
        ):
            pass

    assert read_int_from_file(join_path(tmp_path, "0")) == 2
    assert read_int_from_file(join_path(tmp_path, "1")) == 1
    assert read_int_from_file(join_path(tmp_path, "2")) == 1


def test_map_unordered_stragglers(tmp_path):
    # TODO: run a test like this using a cloud executor
    # Reason: this test passes, but lithops local mode only runs one job at a time,
    # so it actually waits for the first job to finish before running the second one.

    config = {"lithops": {"log_level": "DEBUG"}}
    with LocalhostExecutor(config=config) as executor:
        for _ in map_unordered(
            executor,
            partial(sleep_on_first_invocation, tmp_path),
            range(10),
            use_backups=True,
        ):
            pass

    assert read_int_from_file(join_path(tmp_path, "0")) == 2
    for i in range(1, 10):
        assert read_int_from_file(join_path(tmp_path, f"{i}")) == 1
