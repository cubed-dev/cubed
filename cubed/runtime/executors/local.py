import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, Sequence

import cloudpickle
import psutil
from networkx import MultiDiGraph
from tenacity import Retrying, stop_after_attempt

from cubed.runtime.asyncio import async_map_dag
from cubed.runtime.backup import use_backups_default
from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback, CubedPipeline, DagExecutor, TaskEndEvent
from cubed.runtime.utils import (
    asyncio_run,
    execution_stats,
    execution_timing,
    handle_operation_start_callbacks,
    profile_memray,
)
from cubed.spec import Spec


def exec_stage_func(input, func=None, config=None, name=None, compute_id=None):
    return func(input, config=config)


class SingleThreadedExecutor(DagExecutor):
    """The default execution engine that runs tasks sequentially uses Python loops."""

    @property
    def name(self) -> str:
        return "single-threaded"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        for name, node in visit_nodes(dag, resume=resume):
            handle_operation_start_callbacks(callbacks, name)
            pipeline: CubedPipeline = node["pipeline"]
            for m in pipeline.mappable:
                result = exec_stage_func(
                    m,
                    pipeline.function,
                    config=pipeline.config,
                    name=name,
                    compute_id=compute_id,
                )
                if callbacks is not None:
                    event = TaskEndEvent(name=name, result=result)
                    [callback.on_task_end(event) for callback in callbacks]


@execution_timing
def run_func_threads(input, func=None, config=None, name=None, compute_id=None):
    return func(input, config=config)


@profile_memray
@execution_stats
def run_func_processes(input, func=None, config=None, name=None, compute_id=None):
    return func(input, config=config)


def unpickle_and_call(f, inp, **kwargs):
    import cloudpickle

    f = cloudpickle.loads(f)
    inp = cloudpickle.loads(inp)
    kwargs = {k: cloudpickle.loads(v) for k, v in kwargs.items()}
    return f(inp, **kwargs)


def check_runtime_memory(spec, max_workers):
    allowed_mem = spec.allowed_mem if spec is not None else None
    total_mem = psutil.virtual_memory().total
    if allowed_mem is not None:
        if total_mem < allowed_mem * max_workers:
            raise ValueError(
                f"Total memory on machine ({total_mem}) is less than allowed_mem * max_workers ({allowed_mem} * {max_workers} = {allowed_mem * max_workers})"
            )


def threads_create_futures_func(
    concurrent_executor, function: Callable[..., Any], retries: int = 2
):
    if retries != 0:
        retryer = Retrying(reraise=True, stop=stop_after_attempt(retries + 1))
        function = partial(retryer, function)

    def create_futures_func(input, **kwargs):
        return [
            (
                i,
                asyncio.wrap_future(concurrent_executor.submit(function, i, **kwargs)),
            )
            for i in input
        ]

    return create_futures_func


class ThreadsExecutor(DagExecutor):
    """An execution engine that uses Python asyncio."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # Tell NumPy to use a single thread
        # from https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    @property
    def name(self) -> str:
        return "threads"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio_run(
            self._async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )

    async def _async_execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_arrays_in_parallel: Optional[bool] = None,
        **kwargs,
    ) -> None:
        max_workers = kwargs.pop("max_workers", os.cpu_count())
        if spec is not None:
            check_runtime_memory(spec, max_workers)
            if "use_backups" not in kwargs and use_backups_default(spec):
                kwargs["use_backups"] = True

        concurrent_executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            create_futures_func = threads_create_futures_func(
                concurrent_executor, run_func_threads, kwargs.pop("retries", 2)
            )
            await async_map_dag(
                create_futures_func,
                dag=dag,
                callbacks=callbacks,
                resume=resume,
                compute_arrays_in_parallel=compute_arrays_in_parallel,
                **kwargs,
            )
        finally:
            # don't wait for any cancelled tasks
            concurrent_executor.shutdown(wait=False)


def processes_create_futures_func(concurrent_executor, function: Callable[..., Any]):
    def create_futures_func(input, **kwargs):
        # Pickle the function, args, and kwargs using cloudpickle.
        # They will be unpickled by unpickle_and_call.
        pickled_kwargs = {k: cloudpickle.dumps(v) for k, v in kwargs.items()}
        return [
            (
                i,
                asyncio.wrap_future(
                    concurrent_executor.submit(
                        unpickle_and_call,
                        cloudpickle.dumps(function),
                        cloudpickle.dumps(i),
                        **pickled_kwargs,
                    )
                ),
            )
            for i in input
        ]

    return create_futures_func


class ProcessesExecutor(DagExecutor):
    """An execution engine that uses local processes."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # Tell NumPy to use a single thread
        # from https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    @property
    def name(self) -> str:
        return "processes"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio_run(
            self._async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )

    async def _async_execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_arrays_in_parallel: Optional[bool] = None,
        **kwargs,
    ) -> None:
        max_workers = kwargs.pop("max_workers", os.cpu_count())
        if spec is not None:
            check_runtime_memory(spec, max_workers)
            if "use_backups" not in kwargs and use_backups_default(spec):
                kwargs["use_backups"] = True

        use_processes = kwargs.pop("use_processes", True)
        if isinstance(use_processes, str):
            context = multiprocessing.get_context(use_processes)
        else:
            context = multiprocessing.get_context("spawn")

        # max_tasks_per_child is only supported from Python 3.11
        max_tasks_per_child = kwargs.pop("max_tasks_per_child", None)
        if max_tasks_per_child is None:
            concurrent_executor = ProcessPoolExecutor(
                max_workers=max_workers, mp_context=context
            )
        else:
            concurrent_executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=context,
                max_tasks_per_child=max_tasks_per_child,
            )
        try:
            create_futures_func = processes_create_futures_func(
                concurrent_executor, run_func_processes
            )
            await async_map_dag(
                create_futures_func,
                dag=dag,
                callbacks=callbacks,
                resume=resume,
                compute_arrays_in_parallel=compute_arrays_in_parallel,
                **kwargs,
            )
        finally:
            # don't wait for any cancelled tasks
            concurrent_executor.shutdown(wait=False)
