import asyncio
import time
from multiprocessing.dummy import Pool
from typing import Iterator

import pytest

from cubed.io import map_nested_async, map_nested_concurrent


@pytest.fixture()
def thread_pool():
    pool = Pool(2)
    yield pool
    pool.terminate()


def test_map_nested_concurrent_lists(thread_pool):
    def inc(x):
        print("calling inc", x)
        # time.sleep(2)
        print("DONE calling inc", x)
        return x + 1

    assert map_nested_concurrent(inc, [1, 2], thread_pool) == [2, 3]
    assert map_nested_concurrent(inc, [[1, 2]], thread_pool) == [[2, 3]]
    assert map_nested_concurrent(inc, [[1, 2], [3, 4]], thread_pool) == [[2, 3], [4, 5]]


count = 0


def inc(x):
    global count
    count = count + 1
    print("calling inc", x)
    # time.sleep(x)
    print("DONE calling inc", x)
    return x + 1


def test_map_nested_concurrent_iterators(thread_pool):
    # same tests as test_map_nested_concurrent_lists, but use a counter to check that iterators are advanced at correct points
    global count

    out = map_nested_concurrent(inc, iter([1, 2]), thread_pool)
    assert isinstance(out, Iterator)
    assert count == 0
    assert next(out) == 2
    assert count == 2  # has read 2 already
    assert next(out) == 3
    assert count == 2

    # reset count
    count = 0

    out = map_nested_concurrent(inc, [iter([1, 2])], thread_pool)
    assert isinstance(out, list)
    assert count == 0
    assert len(out) == 1
    out = out[0]
    assert isinstance(out, Iterator)
    assert count == 0
    assert next(out) == 2
    assert count == 2  # has read 2 already
    assert next(out) == 3
    assert count == 2

    # reset count
    count = 0

    out = map_nested_concurrent(inc, [iter([1, 2]), iter([3, 4])], thread_pool)
    assert isinstance(out, list)
    assert count == 0
    assert len(out) == 2
    out0 = out[0]
    assert isinstance(out0, Iterator)
    assert count == 0
    assert next(out0) == 2
    # note we can't assert precisely here since imap does not have back pressure
    assert count > 1  # has read 2 (and maybe more) already
    assert next(out0) == 3
    out1 = out[1]
    assert isinstance(out1, Iterator)
    assert next(out1) == 4
    assert next(out1) == 5
    assert count == 4


def test_map_nested_async_lists():
    asyncio.run(run_test_map_nested_async_lists())


async def run_test_map_nested_async_lists():
    async def inc(x):
        print("calling inc", x)
        await asyncio.sleep(2)
        print("DONE calling inc", x)
        return x + 1

    out = map_nested_async(inc, [1, 2, 3], task_limit=2)

    async with out.stream() as streamer:
        async for z in streamer:
            print(z)
