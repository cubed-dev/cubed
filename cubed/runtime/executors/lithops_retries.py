from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lithops import FunctionExecutor
from lithops.future import ResponseFuture
from lithops.wait import ALL_COMPLETED, ALWAYS, ANY_COMPLETED
from six import reraise


class RetryingFuture:
    """
    A wrapper around Lithops `ResponseFuture` that takes care of retries.
    """

    def __init__(
        self,
        response_future: ResponseFuture,
        map_function: Callable,
        input: Any,
        map_kwargs: Any = None,
        retries: Optional[int] = None,
    ):
        self.response_future = response_future
        self.map_function = map_function
        self.input = input
        self.map_kwargs = map_kwargs or {}
        self.retries = retries or 0
        self.failure_count = 0
        self.cancelled = False

    def _inc_failure_count(self):
        self.failure_count += 1

    def _should_retry(self):
        return not self.cancelled and self.failure_count <= self.retries

    def _retry(self, function_executor: FunctionExecutor):
        inputs = [self.input]
        futures_list = function_executor.map(
            self.map_function, inputs, **self.map_kwargs
        )
        self.response_future = futures_list[0]

    def cancel(self):
        # cancelling will prevent any further retries, but won't affect any running tasks
        self.cancelled = True

    @property
    def done(self):
        return self.response_future.done

    @property
    def error(self):
        return self.response_future.error

    @property
    def _exception(self):
        return self.response_future._exception

    @property
    def stats(self):
        return self.response_future.stats

    def status(
        self,
        throw_except: bool = True,
        internal_storage: Any = None,
        check_only: bool = False,
    ):
        stat = self.response_future.status(
            throw_except=throw_except,
            internal_storage=internal_storage,
            check_only=check_only,
        )
        if self.response_future.error:
            reraise(*self.response_future._exception)
        return stat

    def result(self, throw_except: bool = True, internal_storage: Any = None):
        res = self.response_future.result(
            throw_except=throw_except, internal_storage=internal_storage
        )
        if self.response_future.error:
            reraise(*self.response_future._exception)
        return res


def map_with_retries(
    function_executor: FunctionExecutor,
    map_function: Callable,
    map_iterdata: List[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]],
    timeout: Optional[int] = None,
    include_modules: Optional[List[str]] = [],
    retries: Optional[int] = None,
) -> List[RetryingFuture]:
    """
    A generalisation of Lithops `map`, with retries.
    """

    inputs = list(map_iterdata)
    futures_list = function_executor.map(
        map_function, inputs, timeout=timeout, include_modules=include_modules
    )
    return [
        RetryingFuture(
            f,
            map_function=map_function,
            input=i,
            map_kwargs=dict(timeout=timeout, include_modules=include_modules),
            retries=retries,
        )
        for i, f in zip(inputs, futures_list)
    ]


def wait_with_retries(
    function_executor: FunctionExecutor,
    fs: List[RetryingFuture],
    throw_except: Optional[bool] = True,
    return_when: Optional[Any] = ALL_COMPLETED,
    show_progressbar: Optional[bool] = True,
) -> Tuple[List[RetryingFuture], List[RetryingFuture]]:
    """
    A generalisation of Lithops `wait`, with retries.
    """

    lookup = {f.response_future: f for f in fs}

    while True:
        response_futures = [f.response_future for f in fs]

        done, pending = function_executor.wait(
            response_futures,
            throw_except=throw_except,
            return_when=return_when,
            show_progressbar=show_progressbar,
        )

        retrying_done = []
        retrying_pending = [lookup[response_future] for response_future in pending]
        for response_future in done:
            retrying_future = lookup[response_future]
            if response_future.error:
                retrying_future._inc_failure_count()
                if retrying_future._should_retry():
                    retrying_future._retry(function_executor)
                    # put back into pending since we are retrying this input
                    retrying_pending.append(retrying_future)
                    lookup[retrying_future.response_future] = retrying_future
                else:
                    retrying_done.append(retrying_future)
            else:
                retrying_done.append(retrying_future)

        if return_when == ALWAYS:
            break
        elif return_when == ANY_COMPLETED and len(retrying_done) > 0:
            break
        elif return_when == ALL_COMPLETED and len(retrying_pending) == 0:
            break

    return retrying_done, retrying_pending
