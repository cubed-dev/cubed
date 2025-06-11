import asyncio
import copy
import time
from asyncio import Future
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from aiostream import stream
from aiostream.core import Stream
from networkx import MultiDiGraph

from cubed.runtime.backup import should_launch_backup
from cubed.runtime.pipeline import visit_node_generations, visit_nodes
from cubed.runtime.types import Callback, CubedPipeline
from cubed.runtime.utils import (
    batched,
    handle_callbacks,
    handle_operation_start_callbacks,
)


async def async_map_unordered(
    create_futures_func: Callable[..., List[Tuple[Any, Future]]],
    input: Iterable[Any],
    use_backups: bool = False,
    create_backup_futures_func: Optional[
        Callable[..., List[Tuple[Any, Future]]]
    ] = None,
    batch_size: Optional[int] = None,
    return_stats: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    """
    Asynchronous parallel map over an iterable input, with support for backups and batching.
    """

    if create_backup_futures_func is None:
        create_backup_futures_func = create_futures_func

    if batch_size is None:
        inputs = input
    else:
        input_batches = batched(input, batch_size)
        inputs = next(input_batches)

    task_create_tstamp = time.time()
    tasks = {task: i for i, task in create_futures_func(inputs, name=name, **kwargs)}
    pending = set(tasks.keys())
    t = time.monotonic()
    start_times = {f: t for f in pending}
    end_times = {}
    backups: Dict[asyncio.Future, asyncio.Future] = {}

    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED, timeout=2
        )
        for task in finished:
            # TODO: use exception groups in Python 3.11 to handle case of multiple task exceptions
            if task.exception():
                # if the task has a backup that is not done, or is done with no exception, then don't raise this exception
                backup = backups.get(task, None)
                if backup:
                    if not backup.done() or not backup.exception():
                        continue
                raise task.exception()  # type: ignore
            end_times[task] = time.monotonic()
            if return_stats:
                result, stats = task.result()
                if name is not None:
                    stats["name"] = name
                stats["task_create_tstamp"] = task_create_tstamp
                yield result, stats
            else:
                yield task.result()

            # remove any backup task
            if use_backups:
                backup = backups.get(task, None)
                if backup:
                    if backup in pending:
                        pending.remove(backup)
                    del backups[task]
                    del backups[backup]
                    backup.cancel()

        if use_backups:
            now = time.monotonic()
            for task in copy.copy(pending):
                if task not in backups and should_launch_backup(
                    task, now, start_times, end_times
                ):
                    # launch backup task
                    i = tasks[task]
                    print(f"Launching backup task for input {i} at time {now}")
                    i, new_task = create_backup_futures_func([i], **kwargs)[0]
                    tasks[new_task] = i
                    start_times[new_task] = time.monotonic()
                    pending.add(new_task)
                    backups[task] = new_task
                    backups[new_task] = task

        if batch_size is not None and len(pending) < batch_size:
            inputs = next(input_batches, None)  # type: ignore
            if inputs is not None:
                new_tasks = {
                    task: i for i, task in create_futures_func(inputs, **kwargs)
                }
                tasks.update(new_tasks)
                pending.update(new_tasks.keys())
                t = time.monotonic()
                start_times = {f: t for f in new_tasks.keys()}


async def async_map_dag(
    create_futures_func: Callable,
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    resume: Optional[bool] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    **kwargs,
) -> None:
    """
    Asynchronous parallel map over multiple pipelines from a DAG, with support for backups and batching.
    """
    if not compute_arrays_in_parallel:
        # run one pipeline at a time
        for name, node in visit_nodes(dag, resume=resume):
            handle_operation_start_callbacks(callbacks, name)
            st = pipeline_to_stream(
                create_futures_func, name, node["pipeline"], **kwargs
            )
            async with st.stream() as streamer:
                async for result, stats in streamer:
                    handle_callbacks(callbacks, result, stats)
    else:
        for gen in visit_node_generations(dag, resume=resume):
            # run pipelines in the same topological generation in parallel by merging their streams
            streams = [
                pipeline_to_stream(
                    create_futures_func, name, node["pipeline"], **kwargs
                )
                for name, node in gen
            ]
            merged_stream = stream.merge(*streams)
            async with merged_stream.stream() as streamer:
                async for result, stats in streamer:
                    handle_callbacks(callbacks, result, stats)


def pipeline_to_stream(
    create_futures_func: Callable,
    name: str,
    pipeline: CubedPipeline,
    **kwargs,
) -> Stream:
    """
    Turn a pipeline into an asynchronous stream of results.
    """
    return stream.iterate(
        async_map_unordered(
            create_futures_func,
            pipeline.mappable,
            return_stats=True,
            name=name,
            func=pipeline.function,
            config=pipeline.config,
            **kwargs,
        )
    )
