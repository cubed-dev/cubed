import asyncio
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Sequence

from aiostream import stream
from aiostream.core import Stream
from networkx import MultiDiGraph
from tenacity import Retrying, stop_after_attempt

from cubed.core.array import Callback, Spec
from cubed.core.plan import visit_node_generations, visit_nodes
from cubed.primitive.types import CubedPipeline
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execution_stats, handle_callbacks


@execution_stats
def run_func(input, func=None, config=None, name=None):
    print(f"{name}: running on {input}")
    result = func(input, config=config)
    return result


async def map_unordered(
    concurrent_executor: Executor,
    function: Callable[..., Any],
    input: Iterable[Any],
    retries: int = 2,
    use_backups: bool = False,
    return_stats: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    if name is not None:
        print(f"{name}: running map_unordered")
    if retries == 0:
        retrying_function = function
    else:
        retryer = Retrying(reraise=True, stop=stop_after_attempt(retries + 1))
        retrying_function = partial(retryer, function)

    task_create_tstamp = time.time()
    tasks = {
        asyncio.wrap_future(
            concurrent_executor.submit(retrying_function, i, **kwargs)
        ): i
        for i in input
    }
    pending = set(tasks.keys())

    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED, timeout=2
        )
        for task in finished:
            # TODO: use exception groups in Python 3.11 to handle case of multiple task exceptions
            if task.exception():
                raise task.exception()
            if return_stats:
                result, stats = task.result()
                if name is not None:
                    stats["array_name"] = name
                stats["task_create_tstamp"] = task_create_tstamp
                yield result, stats
            else:
                yield task.result()


def pipeline_to_stream(
    concurrent_executor: Executor, name: str, pipeline: CubedPipeline, **kwargs
) -> Stream:
    it = stream.iterate(
        [
            partial(
                map_unordered,
                concurrent_executor,
                run_func,
                pipeline.mappable,
                return_stats=True,
                name=name,
                func=pipeline.function,
                config=pipeline.config,
                **kwargs,
            )
        ]
    )
    # concat stages, running only one stage at a time
    return stream.concatmap(it, lambda f: f(), task_limit=1)


async def async_execute_dag(
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    array_names: Optional[Sequence[str]] = None,
    resume: Optional[bool] = None,
    spec: Optional[Spec] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    **kwargs,
) -> None:
    with ThreadPoolExecutor() as concurrent_executor:
        if not compute_arrays_in_parallel:
            # run one pipeline at a time
            for name, node in visit_nodes(dag, resume=resume):
                st = pipeline_to_stream(
                    concurrent_executor, name, node["pipeline"], **kwargs
                )
                async with st.stream() as streamer:
                    async for _, stats in streamer:
                        handle_callbacks(callbacks, stats)
        else:
            for gen in visit_node_generations(dag, resume=resume):
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


class AsyncPythonDagExecutor(DagExecutor):
    """An execution engine that uses Python asyncio."""

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        array_names: Optional[Sequence[str]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        **kwargs,
    ) -> None:
        asyncio.run(
            async_execute_dag(
                dag,
                callbacks=callbacks,
                array_names=array_names,
                resume=resume,
                spec=spec,
                **kwargs,
            )
        )
