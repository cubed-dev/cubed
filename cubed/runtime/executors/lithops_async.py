import asyncio
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
    Union,
)

from aiostream import stream
from aiostream.core import Stream
from lithops.executors import FunctionExecutor
from lithops.wait import ALWAYS, ANY_COMPLETED
from networkx import MultiDiGraph

from cubed.core.array import Callback, Spec
from cubed.core.plan import visit_node_generations, visit_nodes
from cubed.primitive.types import CubedPipeline
from cubed.runtime.executors.asyncio import async_map_unordered
from cubed.runtime.executors.lithops_async_wrapper import AsyncFunctionExecutorWrapper
from cubed.runtime.executors.lithops_retries import RetryingFunctionExecutor
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execution_stats, handle_callbacks


@execution_stats
def run_func(input, func=None, config=None, name=None):
    result = func(input, config=config)
    return result


async def map_unordered(
    executor: AsyncFunctionExecutorWrapper,
    map_function: Callable[..., Any],
    map_iterdata: Iterable[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]],
    timeout: Optional[int] = None,
    retries: int = 2,
    use_backups: bool = False,
    return_stats: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    def create_futures_func(input, **kwargs):
        # TODO: lithops stats

        # can't use functools.partial here as we get an error in lithops
        # also, lithops extra_args doesn't work for this case
        partial_map_function = lambda x: map_function(x, **kwargs)

        input = list(input)
        return list(
            zip(
                input,
                executor.map(
                    partial_map_function, input, timeout=timeout, retries=retries
                ),
            )
        )

    def create_backup_futures_func(input, **kwargs):
        # can't use functools.partial here as we get an error in lithops
        # also, lithops extra_args doesn't work for this case
        partial_map_function = lambda x: map_function(x, **kwargs)

        input = list(input)
        return list(
            zip(
                input,
                executor.map(partial_map_function, input, timeout=timeout, retries=0),
            )
        )

    async for result in async_map_unordered(
        create_futures_func,
        map_iterdata,
        use_backups=use_backups,
        create_backup_futures_func=create_backup_futures_func,
        return_stats=return_stats,
        name=name,
        **kwargs,
    ):
        yield result


def pipeline_to_stream(
    executor: AsyncFunctionExecutorWrapper, name: str, pipeline: CubedPipeline, **kwargs
) -> Stream:
    return stream.iterate(
        map_unordered(
            executor,
            run_func,
            pipeline.mappable,
            return_stats=True,
            name=name,
            func=pipeline.function,
            config=pipeline.config,
            **kwargs,
        )
    )


async def async_execute_dag(
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    array_names: Optional[Sequence[str]] = None,
    resume: Optional[bool] = None,
    spec: Optional[Spec] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    **kwargs,
) -> None:
    use_backups = kwargs.pop("use_backups", False)
    allowed_mem = spec.allowed_mem if spec is not None else None
    lithops_executor = FunctionExecutor(**kwargs)
    runtime_memory_mb = lithops_executor.config[lithops_executor.backend].get(
        "runtime_memory", None
    )
    if runtime_memory_mb is not None and allowed_mem is not None:
        runtime_memory = runtime_memory_mb * 1_000_000
        if runtime_memory < allowed_mem:
            raise ValueError(
                f"Runtime memory ({runtime_memory}) is less than allowed_mem ({allowed_mem})"
            )
    return_when = ALWAYS if use_backups else ANY_COMPLETED
    with AsyncFunctionExecutorWrapper(
        RetryingFunctionExecutor(lithops_executor), return_when
    ) as executor:
        if not compute_arrays_in_parallel:
            # run one pipeline at a time
            for name, node in visit_nodes(dag, resume=resume):
                st = pipeline_to_stream(executor, name, node["pipeline"])
                async with st.stream() as streamer:
                    async for _, stats in streamer:
                        handle_callbacks(callbacks, stats)
        else:
            for gen in visit_node_generations(dag, resume=resume):
                # run pipelines in the same topological generation in parallel by merging their streams
                streams = [
                    pipeline_to_stream(executor, name, node["pipeline"])
                    for name, node in gen
                ]
                merged_stream = stream.merge(*streams)
                async with merged_stream.stream() as streamer:
                    async for _, stats in streamer:
                        handle_callbacks(callbacks, stats)


class AsyncLithopsExecutor(DagExecutor):
    """An execution engine that uses Lithops via the async wrapper."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        array_names: Optional[Sequence[str]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio.run(
            async_execute_dag(
                dag,
                callbacks=callbacks,
                array_names=array_names,
                resume=resume,
                spec=spec,
                **merged_kwargs,
            )
        )
