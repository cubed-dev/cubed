import asyncio
import multiprocessing
import os
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Sequence

import cloudpickle
import psutil
from aiostream import stream
from aiostream.core import Stream
from networkx import MultiDiGraph
from tenacity import Retrying, stop_after_attempt

from cubed.runtime.backup import use_backups_default
from cubed.runtime.executors.asyncio import async_map_unordered
from cubed.runtime.pipeline import visit_node_generations, visit_nodes
from cubed.runtime.types import Callback, CubedPipeline, DagExecutor, TaskEndEvent
from cubed.runtime.utils import (
    asyncio_run,
    execution_stats,
    execution_timing,
    handle_callbacks,
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


async def map_unordered(
    concurrent_executor: Executor,
    function: Callable[..., Any],
    input: Iterable[Any],
    retries: int = 2,
    use_backups: bool = False,
    batch_size: Optional[int] = None,
    return_stats: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    if retries == 0:
        retrying_function = function
    else:
        if isinstance(concurrent_executor, ProcessPoolExecutor):
            raise NotImplementedError("Retries not supported for ProcessPoolExecutor")
        retryer = Retrying(reraise=True, stop=stop_after_attempt(retries + 1))
        retrying_function = partial(retryer, function)

    def create_futures_func(input, **kwargs):
        return [
            (
                i,
                asyncio.wrap_future(
                    concurrent_executor.submit(retrying_function, i, **kwargs)
                ),
            )
            for i in input
        ]

    def create_futures_func_multiprocessing(input, **kwargs):
        # Pickle the function, args, and kwargs using cloudpickle.
        # They will be unpickled by unpickle_and_call.
        pickled_kwargs = {k: cloudpickle.dumps(v) for k, v in kwargs.items()}
        return [
            (
                i,
                asyncio.wrap_future(
                    concurrent_executor.submit(
                        unpickle_and_call,
                        cloudpickle.dumps(retrying_function),
                        cloudpickle.dumps(i),
                        **pickled_kwargs,
                    )
                ),
            )
            for i in input
        ]

    if isinstance(concurrent_executor, ProcessPoolExecutor):
        create_futures = create_futures_func_multiprocessing
    else:
        create_futures = create_futures_func
    async for result in async_map_unordered(
        create_futures,
        input,
        use_backups=use_backups,
        batch_size=batch_size,
        return_stats=return_stats,
        name=name,
        **kwargs,
    ):
        yield result


def pipeline_to_stream(
    concurrent_executor: Executor,
    run_func: Callable,
    name: str,
    pipeline: CubedPipeline,
    **kwargs,
) -> Stream:
    return stream.iterate(
        map_unordered(
            concurrent_executor,
            run_func,
            pipeline.mappable,
            return_stats=True,
            name=name,
            func=pipeline.function,
            config=pipeline.config,
            **kwargs,
        )
    )


def check_runtime_memory(spec, max_workers):
    allowed_mem = spec.allowed_mem if spec is not None else None
    total_mem = psutil.virtual_memory().total
    if allowed_mem is not None:
        if total_mem < allowed_mem * max_workers:
            raise ValueError(
                f"Total memory on machine ({total_mem}) is less than allowed_mem * max_workers ({allowed_mem} * {max_workers} = {allowed_mem * max_workers})"
            )


async def async_execute_dag(
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    resume: Optional[bool] = None,
    spec: Optional[Spec] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    **kwargs,
) -> None:
    concurrent_executor: Executor
    max_workers = kwargs.pop("max_workers", os.cpu_count())
    use_processes = kwargs.pop("use_processes", False)
    if spec is not None:
        check_runtime_memory(spec, max_workers)
        if "use_backups" not in kwargs and use_backups_default(spec):
            kwargs["use_backups"] = True
    if use_processes:
        max_tasks_per_child = kwargs.pop("max_tasks_per_child", None)
        if isinstance(use_processes, str):
            context = multiprocessing.get_context(use_processes)
        else:
            context = multiprocessing.get_context("spawn")
        # max_tasks_per_child is only supported from Python 3.11
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
        run_func = run_func_processes
    else:
        concurrent_executor = ThreadPoolExecutor(max_workers=max_workers)
        run_func = run_func_threads
    try:
        if not compute_arrays_in_parallel:
            # run one pipeline at a time
            for name, node in visit_nodes(dag, resume=resume):
                handle_operation_start_callbacks(callbacks, name)
                st = pipeline_to_stream(
                    concurrent_executor, run_func, name, node["pipeline"], **kwargs
                )
                async with st.stream() as streamer:
                    async for result, stats in streamer:
                        handle_callbacks(callbacks, result, stats)
        else:
            for gen in visit_node_generations(dag, resume=resume):
                # run pipelines in the same topological generation in parallel by merging their streams
                streams = [
                    pipeline_to_stream(
                        concurrent_executor, run_func, name, node["pipeline"], **kwargs
                    )
                    for name, node in gen
                ]
                merged_stream = stream.merge(*streams)
                async with merged_stream.stream() as streamer:
                    async for result, stats in streamer:
                        handle_callbacks(callbacks, result, stats)

    finally:
        # don't wait for any cancelled tasks
        concurrent_executor.shutdown(wait=False)


class ThreadsExecutor(DagExecutor):
    """An execution engine that uses Python asyncio."""

    def __init__(self, **kwargs):
        self.kwargs = {**kwargs, **dict(use_processes=False)}

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
            async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )


class ProcessesExecutor(DagExecutor):
    """An execution engine that uses local processes."""

    def __init__(self, **kwargs):
        self.kwargs = {**kwargs, **dict(retries=0, use_processes=True)}

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
            async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )
