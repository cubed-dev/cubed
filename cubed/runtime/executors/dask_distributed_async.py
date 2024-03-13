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
from dask.distributed import Client
from networkx import MultiDiGraph

from cubed.runtime.executors.asyncio import async_map_unordered
from cubed.runtime.pipeline import visit_node_generations, visit_nodes
from cubed.runtime.types import Callback, CubedPipeline, DagExecutor
from cubed.runtime.utils import (
    execution_stats,
    gensym,
    handle_callbacks,
    handle_operation_start_callbacks,
)
from cubed.spec import Spec


# note we can't call `pipeline_func` just `func` here as it clashes with `dask.distributed.Client.map``
@execution_stats
def run_func(input, pipeline_func=None, config=None, name=None, compute_id=None):
    result = pipeline_func(input, config=config)
    return result


async def map_unordered(
    client: Client,
    map_function: Callable[..., Any],
    map_iterdata: Iterable[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]],
    retries: int = 2,
    use_backups: bool = True,
    batch_size: Optional[int] = None,
    return_stats: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> AsyncIterator[Any]:
    def create_futures_func(input, **kwargs):
        input = list(input)  # dask expects a sequence (it calls `len` on it)
        key = name or gensym("map")
        key = key.replace("-", "_")  # otherwise array number is not shown on dashboard
        return [
            (i, asyncio.ensure_future(f))
            for i, f in zip(
                input,
                client.map(map_function, input, key=key, retries=retries, **kwargs),
            )
        ]

    def create_backup_futures_func(input, **kwargs):
        input = list(input)  # dask expects a sequence (it calls `len` on it)
        key = name or gensym("backup")
        key = key.replace("-", "_")  # otherwise array number is not shown on dashboard
        return [
            (i, asyncio.ensure_future(f))
            for i, f in zip(input, client.map(map_function, input, key=key, **kwargs))
        ]

    async for result in async_map_unordered(
        create_futures_func,
        map_iterdata,
        use_backups=use_backups,
        create_backup_futures_func=create_backup_futures_func,
        batch_size=batch_size,
        return_stats=return_stats,
        name=name,
        **kwargs,
    ):
        yield result


def pipeline_to_stream(
    client: Client, name: str, pipeline: CubedPipeline, **kwargs
) -> Stream:
    return stream.iterate(
        map_unordered(
            client,
            run_func,
            pipeline.mappable,
            return_stats=True,
            name=name,
            pipeline_func=pipeline.function,
            config=pipeline.config,
            **kwargs,
        )
    )


def check_runtime_memory(spec, client):
    allowed_mem = spec.allowed_mem if spec is not None else None
    scheduler_info = client.scheduler_info()
    workers = scheduler_info.get("workers")
    if workers is None or len(workers) == 0:
        raise ValueError("Cluster has no workers running")
    runtime_memory = min(w["memory_limit"] // w["nthreads"] for w in workers.values())
    if allowed_mem is not None:
        if runtime_memory < allowed_mem:
            raise ValueError(
                f"Runtime memory ({runtime_memory}) is less than allowed_mem ({allowed_mem})"
            )


async def async_execute_dag(
    dag: MultiDiGraph,
    callbacks: Optional[Sequence[Callback]] = None,
    resume: Optional[bool] = None,
    spec: Optional[Spec] = None,
    compute_arrays_in_parallel: Optional[bool] = None,
    compute_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    compute_kwargs = compute_kwargs or {}
    async with Client(asynchronous=True, **compute_kwargs) as client:
        if spec is not None:
            check_runtime_memory(spec, client)
        if not compute_arrays_in_parallel:
            # run one pipeline at a time
            for name, node in visit_nodes(dag, resume=resume):
                handle_operation_start_callbacks(callbacks, name)
                st = pipeline_to_stream(client, name, node["pipeline"], **kwargs)
                async with st.stream() as streamer:
                    async for _, stats in streamer:
                        handle_callbacks(callbacks, stats)
        else:
            for gen in visit_node_generations(dag, resume=resume):
                # run pipelines in the same topological generation in parallel by merging their streams
                streams = [
                    pipeline_to_stream(client, name, node["pipeline"], **kwargs)
                    for name, node in gen
                ]
                merged_stream = stream.merge(*streams)
                async with merged_stream.stream() as streamer:
                    async for _, stats in streamer:
                        handle_callbacks(callbacks, stats)


class AsyncDaskDistributedExecutor(DagExecutor):
    """An execution engine that uses Dask Distributed's async API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "dask"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {**self.kwargs, **kwargs}
        asyncio.run(
            async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_kwargs=compute_kwargs,
                compute_id=compute_id,
                **merged_kwargs,
            )
        )
