import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from aiostream import stream
from tenacity import retry, stop_after_attempt

from cubed.core.array import TaskEndEvent
from cubed.core.plan import visit_node_generations
from cubed.runtime.types import DagExecutor
from cubed.utils import peak_memory


def execution_stats(func):
    """Measure timing information and peak memory usage of a function call."""

    def wrapper(*args, **kwargs):
        peak_memory_start = peak_memory()
        function_start_tstamp = time.time()
        result = func(*args, **kwargs)
        function_end_tstamp = time.time()
        peak_memory_end = peak_memory()
        return result, dict(
            function_start_tstamp=function_start_tstamp,
            function_end_tstamp=function_end_tstamp,
            peak_memory_start=peak_memory_start,
            peak_memory_end=peak_memory_end,
        )

    return wrapper


@retry(stop=stop_after_attempt(3))
@execution_stats
def run_func(input, func=None, config=None, name=None):
    print(f"{name}: running on {input}")
    result = func(input, config=config)
    return result


async def map_unordered(
    concurrent_executor,
    function,
    input,
    max_failures=3,
    use_backups=False,
    **kwargs,
):
    print(f"{kwargs['name']}: running map_unordered")
    tasks = {
        asyncio.wrap_future(concurrent_executor.submit(function, i, **kwargs)): i
        for i in input
    }
    pending = set(tasks.keys())

    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED, timeout=2
        )
        for task in finished:
            if task.exception():
                raise task.exception()
            yield task.result()


async def produce(*args, **kwargs):
    task_create_tstamp = time.time()
    name = kwargs.get("name", None)
    async for result, stats in map_unordered(*args, **kwargs):
        stats["array_name"] = name
        stats["task_create_tstamp"] = task_create_tstamp
        yield result, stats


def pipeline_to_stream(concurrent_executor, name, pipeline, **kwargs):
    if any([stage for stage in pipeline.stages if stage.mappable is None]):
        raise NotImplementedError("All stages must be mappable in pipelines")
    it = stream.iterate(
        [
            partial(
                produce,
                concurrent_executor,
                run_func,
                list(stage.mappable),
                func=stage.function,
                config=pipeline.config,
                name=name,
                **kwargs,
            )
            for stage in pipeline.stages
            if stage.mappable is not None
        ]
    )
    # concat stages, running only one stage at a time
    return stream.concatmap(it, lambda f: f(), task_limit=1)


async def async_execute_dag(dag, callbacks=None, **kwargs):
    with ThreadPoolExecutor() as concurrent_executor:
        for gen in visit_node_generations(dag):
            # run pipelines in the same topological generation in parallel by merging their streams
            streams = [
                pipeline_to_stream(
                    concurrent_executor, name, node["pipeline"], **kwargs
                )
                for name, node in gen
            ]
            merged_stream = stream.merge(*streams)
            async with merged_stream.stream() as streamer:
                async for _, stats in streamer:
                    handle_callbacks(callbacks, stats)


def handle_callbacks(callbacks, stats):
    if callbacks is not None:
        task_result_tstamp = time.time()
        event = TaskEndEvent(
            task_result_tstamp=task_result_tstamp,
            **stats,
        )
        [callback.on_task_end(event) for callback in callbacks]


class AsyncPythonDagExecutor(DagExecutor):
    """An execution engine that uses Python asyncio."""

    def execute_dag(self, dag, callbacks=None, **kwargs):
        asyncio.run(async_execute_dag(dag, callbacks=callbacks, **kwargs))
