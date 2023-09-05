import asyncio
import threading
import time
from functools import partial
from queue import Empty, SimpleQueue

from lithops.wait import ALWAYS

SHUTDOWN_SENTINEL = object()


class AsyncFunctionExecutorWrapper:
    """
    A class that wraps a Lithops `FunctionExecutor` so it can be used with the
    Python asyncio API.

    In particular, `map` returns Python asyncio futures, so they can be awaited
    using the standard asyncio API.

    TODO: usage example
    """

    def __init__(self, executor, background_return_when=ALWAYS):
        self.executor = executor
        self.return_when = background_return_when

    def __enter__(self):
        # enter executor
        self.executor.__enter__()

        # start a queue to send lithops futures and their callbacks to a worker
        self.queue = SimpleQueue()

        # start the worker thread
        self.worker = threading.Thread(
            target=partial(create_worker, self.queue, self.executor, self.return_when)
        )
        self.worker.start()

        return self

    def wrap_futures(self, fs):
        # wrap lithops futures in asyncio futures, and create callbacks for them
        aio_futures, callbacks = _wrap_futures(fs)

        # put on the queue for the worker
        self.queue.put((fs, callbacks))

        # return asyncio futures
        return aio_futures

    def map(self, *args, **kwargs):
        fs = self.executor.map(*args, **kwargs)
        return self.wrap_futures(fs)

    def __exit__(self, exc_type, exc_value, traceback):
        # send a sentinel value to the queue so that the worker exits
        self.queue.put(SHUTDOWN_SENTINEL)
        # shutdown the worker
        self.worker.join()
        # exit executor
        self.executor.__exit__(exc_type, exc_value, traceback)


def create_worker(queue, executor, return_when):
    # note that this does not run in the main thread

    # the worker is responsible for getting any new lithops futures and their callbacks
    # from the queues, and waiting on them using lithops wait

    pending = []
    callbacks = {}

    while True:
        # if there are no pending futures, then block until we get some, or a shutdown signal
        item = queue.get()
        if item is SHUTDOWN_SENTINEL:
            return
        new_futures, new_callbacks = item
        pending.extend(new_futures)
        callbacks.update(new_callbacks)

        # while there are pending futures wait on them using lithops
        while pending:
            finished, pending = executor.wait(
                pending,
                throw_except=False,
                return_when=return_when,
                show_progressbar=False,
            )
            for f in finished:
                callbacks.pop(f)()

            # check the queue for any more futures
            try:
                item = queue.get_nowait()
                if item is SHUTDOWN_SENTINEL:
                    # note that any pending tasks are ignored
                    return
                new_futures, new_callbacks = item
                pending.extend(new_futures)
                callbacks.update(new_callbacks)
            except Empty:
                pass

            # sleep if wait is always returning immediately
            if return_when == ALWAYS:
                time.sleep(1)


# based on https://stackoverflow.com/questions/49350346/how-to-wrap-custom-future-to-use-with-asyncio-in-python
def wrap_future(f):
    loop = asyncio.get_event_loop()
    aio_future = loop.create_future()

    def on_done(*_):
        try:
            result = f.result()
        except Exception as e:
            loop.call_soon_threadsafe(aio_future.set_exception, e)
        else:
            loop.call_soon_threadsafe(aio_future.set_result, result)

    # return the callback since we can't set it on the lithops future
    return aio_future, on_done


def _wrap_futures(fs):
    aio_futures = []
    callbacks = {}
    for f in fs:
        aio_future, callback = wrap_future(f)
        aio_futures.append(aio_future)
        callbacks[f] = callback
    return aio_futures, callbacks
