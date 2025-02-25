import asyncio
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

from cubed.runtime.types import OperationStartEvent, TaskEndEvent
from cubed.utils import is_local_path, join_path, peak_measured_mem

try:
    import memray
except ImportError:
    memray = None  # type: ignore

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


def execution_stats(func):
    """Decorator to measure timing information and peak memory usage of a function call."""

    return partial(execute_with_stats, func)


def execution_timing(func):
    """Decorator to measure timing information of a function call."""

    return partial(execute_with_timing, func)


def execute_with_memray(function, input, **kwargs):
    # only run memray if installed, and only for first input (for operations that run on block locations)
    if (
        memray is not None
        and "context_dir" in kwargs
        and "compute_id" in kwargs
        and isinstance(input, list)
        and all(isinstance(i, int) for i in input)
        and sum(input) == 0
    ):
        print("execute_with_memray", function, input, kwargs)
        context_dir = kwargs["context_dir"]
        print("context_dir", context_dir)
        compute_id = kwargs["compute_id"]
        name = kwargs["name"]
        memray_dir = Path(tempfile.mkdtemp())
        memray_dir.mkdir(parents=True, exist_ok=True)
        memray_file = memray_dir / f"{name}.bin"
        print("memray_file", memray_file)
        try:
            with memray.Tracker(memray_file):
                return function(input, **kwargs)
        finally:
            try:
                copy_from_local(
                    memray_file.as_posix(),
                    join_path(context_dir, f"history/{compute_id}/memray/{name}.bin"),
                )
            except Exception as e:
                # don't fail task if error copying memray file
                print(e)
    else:
        return function(input, **kwargs)


def copy_from_local(local_src: str, remote_dst: str) -> None:
    if is_local_path(remote_dst):
        Path(remote_dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_src, remote_dst)
    else:
        import fsspec

        # dst can be any fsspec filesystem
        fs, _, _ = fsspec.get_fs_token_paths(remote_dst)
        fs.put_file(local_src, remote_dst)


def profile_memray(func):
    """Decorator to profile a function call with memray."""
    return partial(execute_with_memray, func)


def handle_operation_start_callbacks(callbacks, name):
    if callbacks is not None:
        event = OperationStartEvent(name)
        [callback.on_operation_start(event) for callback in callbacks]


def handle_callbacks(callbacks, result, stats):
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
        [callback.on_task_end(event) for callback in callbacks]


# Like asyncio.run(), but works in a Jupyter notebook
# Based on https://stackoverflow.com/a/75341431
def asyncio_run(coro):
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
