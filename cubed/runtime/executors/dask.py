import asyncio
from typing import Any, Callable, Dict, Optional, Sequence

from dask.distributed import Client
from networkx import MultiDiGraph

from cubed.runtime.asyncio import async_map_dag
from cubed.runtime.backup import use_backups_default
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import asyncio_run, execution_stats, gensym
from cubed.spec import Spec


# note we can't call `pipeline_func` just `func` here as it clashes with `dask.distributed.Client.map``
@execution_stats
def run_func(input, pipeline_func=None, config=None, name=None, compute_id=None):
    result = pipeline_func(input, config=config)
    return result


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


def dask_create_futures_func(
    client,
    function: Callable[..., Any],
    name: Optional[str] = None,
    retries: Optional[str] = None,
):
    def create_futures_func(input, **kwargs):
        input = list(input)  # dask expects a sequence (it calls `len` on it)
        key = name or gensym("map")
        key = key.replace("-", "_")  # otherwise array number is not shown on dashboard
        if "func" in kwargs:
            kwargs["pipeline_func"] = kwargs.pop("func")  # rename to avoid clash
        return [
            (i, asyncio.ensure_future(f))
            for i, f in zip(
                input,
                client.map(function, input, key=key, retries=retries, **kwargs),
            )
        ]

    return create_futures_func


class DaskExecutor(DagExecutor):
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
        asyncio_run(
            self._async_execute_dag(
                dag,
                callbacks=callbacks,
                resume=resume,
                spec=spec,
                compute_kwargs=compute_kwargs,
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
        compute_kwargs: Optional[Dict[str, Any]] = None,
        compute_arrays_in_parallel: Optional[bool] = None,
        **kwargs,
    ) -> None:
        compute_kwargs = compute_kwargs or {}
        retries = kwargs.pop("retries", 2)
        name = kwargs.get("name", None)
        async with Client(asynchronous=True, **compute_kwargs) as client:
            if spec is not None:
                check_runtime_memory(spec, client)
                if "use_backups" not in kwargs and use_backups_default(spec):
                    kwargs["use_backups"] = True

            create_futures_func = dask_create_futures_func(
                client, run_func, name, retries=retries
            )
            create_backup_futures_func = dask_create_futures_func(
                client, run_func, name
            )

            await async_map_dag(
                create_futures_func,
                dag=dag,
                callbacks=callbacks,
                resume=resume,
                compute_arrays_in_parallel=compute_arrays_in_parallel,
                create_backup_futures_func=create_backup_futures_func,
                **kwargs,
            )
