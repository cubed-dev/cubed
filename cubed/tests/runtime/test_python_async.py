import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

from cubed.runtime.executors.python_async import map_unordered
from cubed.tests.runtime.utils import (
    fail_on_first_invocation,
    never_fail,
    read_int_from_file,
)
from cubed.utils import join_path


async def run_test(function, input, retries=2):
    with ThreadPoolExecutor() as concurrent_executor:
        async for _ in map_unordered(
            concurrent_executor, function, input, retries=retries
        ):
            pass


def test_map_unordered_no_failures(tmp_path):
    asyncio.run(run_test(function=partial(never_fail, tmp_path), input=range(3)))

    assert read_int_from_file(join_path(tmp_path, "0")) == 1
    assert read_int_from_file(join_path(tmp_path, "1")) == 1
    assert read_int_from_file(join_path(tmp_path, "2")) == 1


def test_map_unordered_recovers_from_failures(tmp_path):
    asyncio.run(
        run_test(function=partial(fail_on_first_invocation, tmp_path), input=range(3))
    )

    assert read_int_from_file(join_path(tmp_path, "0")) == 2
    assert read_int_from_file(join_path(tmp_path, "1")) == 2
    assert read_int_from_file(join_path(tmp_path, "2")) == 2


def test_map_unordered_too_many_failures(tmp_path):
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_test(
                function=partial(fail_on_first_invocation, tmp_path),
                input=range(3),
                retries=0,
            )
        )
