import asyncio
from typing import Optional, Sequence

import ray
from networkx import MultiDiGraph

from cubed.runtime.asyncio import async_map_dag
from cubed.runtime.backup import use_backups_default
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import asyncio_run, execute_with_stats
from cubed.spec import Spec


class RayExecutor(DagExecutor):
    """An execution engine that uses Ray."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "ray"

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

        ray_init = merged_kwargs.pop("ray_init", None)
        if ray_init is not None:
            ray.init(**ray_init)

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
        if spec is not None:
            if "use_backups" not in kwargs and use_backups_default(spec):
                kwargs["use_backups"] = True

        allowed_mem = spec.allowed_mem if spec is not None else 2_000_000_000
        retries = kwargs.pop("retries", 2)

        # note we can define the remote function here (doesn't need to be at top-level), and pass in memory and retries
        @ray.remote(memory=allowed_mem, max_retries=retries, retry_exceptions=True)
        def run_remotely(input, func=None, config=None, name=None, compute_id=None):
            # note we can't use the execution_stat decorator since it doesn't work with ray decorators
            result, stats = execute_with_stats(func, input, config=config)
            return result, stats

        def create_futures_func(input, **kwargs):
            return [
                (i, asyncio.wrap_future(run_remotely.remote(i, **kwargs).future()))
                for i in input
            ]

        await async_map_dag(
            create_futures_func,
            dag=dag,
            callbacks=callbacks,
            resume=resume,
            compute_arrays_in_parallel=compute_arrays_in_parallel,
            **kwargs,
        )
