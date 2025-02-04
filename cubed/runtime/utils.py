import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Coroutine, Dict, Iterable, Optional, TypeVar

from cubed.runtime.types import (
    Callback,
    OperationEndEvent,
    OperationStartEvent,
    TaskEndEvent,
)
from cubed.utils import peak_measured_mem

try:
    import memray
except ImportError:
    memray = None  # type: ignore

T = TypeVar("T")

sym_counter = 0


def gensym(name: str) -> str:
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


def execute_with_stats(function, *args, **kwargs):
    """Invoke function and measure timing information and peak memory usage.

    Returns the result of the function call and a stats dictionary.
    """

    peak_measured_mem_start = peak_measured_mem()
    function_start_tstamp = time.time()
    result = function(*args, **kwargs)
    function_end_tstamp = time.time()
    peak_measured_mem_end = peak_measured_mem()
    return result, dict(
        function_start_tstamp=function_start_tstamp,
        function_end_tstamp=function_end_tstamp,
        peak_measured_mem_start=peak_measured_mem_start,
        peak_measured_mem_end=peak_measured_mem_end,
    )


def execute_with_timing(function, *args, **kwargs):
    """Invoke function and measure timing information.

    Returns the result of the function call and a stats dictionary.
    """

    function_start_tstamp = time.time()
    result = function(*args, **kwargs)
    function_end_tstamp = time.time()
    return result, dict(
        function_start_tstamp=function_start_tstamp,
        function_end_tstamp=function_end_tstamp,
    )


def execution_stats(func, name=None):
    """Decorator to measure timing information and peak memory usage of a function call."""

    def wrapper(*args, **kwargs):
        return execute_with_stats(func, *args, **kwargs)

    wrapper.__name__ = name or func.__name__
    return wrapper


def execution_timing(func):
    """Decorator to measure timing information of a function call."""

    return partial(execute_with_timing, func)


def execute_with_memray(function, input, **kwargs):
    # only run memray if installed, and only for first input (for operations that run on block locations)
    if (
        memray is not None
        and "compute_id" in kwargs
        and isinstance(input, list)
        and all(isinstance(i, int) for i in input)
        and sum(input) == 0
    ):
        compute_id = kwargs["compute_id"]
        name = kwargs["name"]
        memray_dir = Path(f"history/{compute_id}/memray")
        memray_dir.mkdir(parents=True, exist_ok=True)
        cm = memray.Tracker(memray_dir / f"{name}.bin")
    else:
        cm = nullcontext()
    with cm:
        result = result = function(input, **kwargs)
        return result


def profile_memray(func):
    """Decorator to profile a function call with memray."""
    return partial(execute_with_memray, func)


def handle_operation_start_callbacks(
    callbacks: Optional[Iterable[Callback]], name: str
) -> None:
    if callbacks is not None:
        event = OperationStartEvent(name)
        for callback in callbacks:
            callback.on_operation_start(event)


def handle_operation_end_callbacks(callbacks, name) -> None:
    if callbacks is not None:
        event = OperationEndEvent(name)
        for callback in callbacks:
            callback.on_operation_end(event)


def handle_callbacks(
    callbacks: Optional[Iterable[Callback]],
    result: Optional[Any],
    stats: Dict[str, Any],
) -> None:
    """Construct a TaskEndEvent from stats and send to all callbacks."""

    if callbacks is not None:
        if "task_result_tstamp" not in stats:
            task_result_tstamp = time.time()
            event = TaskEndEvent(
                result=result,
                task_result_tstamp=task_result_tstamp,
                **stats,
            )
        else:
            event = TaskEndEvent(result=result, **stats)
        for callback in callbacks:
            callback.on_task_end(event)


def raise_if_computes():
    """Returns a context manager for testing that ``compute`` is not called."""
    from cubed import config

    # note that this is not set on the spec, see compute() in array.py
    return config.set({"executor_name": "raise-if-computes"})


# Like asyncio.run(), but works in a Jupyter notebook
# Based on https://stackoverflow.com/a/75341431
def asyncio_run(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()  # Triggers RuntimeError if no running event loop
    except RuntimeError:
        return asyncio.run(coro)
    else:
        # Create a separate thread so we can block before returning
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()


# this will be in Python 3.12 https://docs.python.org/3.12/library/itertools.html#itertools.batched
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
