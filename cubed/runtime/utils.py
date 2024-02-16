import time
from functools import partial
from itertools import islice

from cubed.runtime.types import OperationStartEvent, TaskEndEvent
from cubed.utils import peak_measured_mem

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


def execution_stats(func):
    """Decorator to measure timing information and peak memory usage of a function call."""

    return partial(execute_with_stats, func)


def handle_operation_start_callbacks(callbacks, name):
    if callbacks is not None:
        event = OperationStartEvent(name)
        [callback.on_operation_start(event) for callback in callbacks]


def handle_callbacks(callbacks, stats):
    """Construct a TaskEndEvent from stats and send to all callbacks."""

    if callbacks is not None:
        if "task_result_tstamp" not in stats:
            task_result_tstamp = time.time()
            event = TaskEndEvent(
                task_result_tstamp=task_result_tstamp,
                **stats,
            )
        else:
            event = TaskEndEvent(**stats)
        [callback.on_task_end(event) for callback in callbacks]


# this will be in Python 3.12 https://docs.python.org/3.12/library/itertools.html#itertools.batched
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
