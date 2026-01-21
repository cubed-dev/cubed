import asyncio
from typing import Any, Callable, Optional, Sequence

from globus_compute_sdk import Client, Executor
from globus_compute_sdk.serialize import CombinedCode
from networkx import MultiDiGraph

from cubed.runtime.asyncio import async_map_dag
from cubed.runtime.backup import use_backups_default
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import asyncio_run
from cubed.spec import Spec


class GlobusComputeExecutor(DagExecutor):
    """An execution engine that uses Globus Compute."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "globus-compute"

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
        if spec is not None:
            if "use_backups" not in kwargs and use_backups_default(spec):
                kwargs["use_backups"] = True

        endpoint_id = kwargs.pop("endpoint_id")
        client = Client(code_serialization_strategy=CombinedCode())
        concurrent_executor = Executor(endpoint_id=endpoint_id, client=client)
        try:
            create_futures_func = globus_compute_create_futures_func(
                concurrent_executor,
                run_func_threads,
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


def globus_compute_create_futures_func(
    concurrent_executor, function: Callable[..., Any]
):
    def create_futures_func(input, **kwargs):
        return [
            (
                i,
                asyncio.wrap_future(concurrent_executor.submit(function, i, **kwargs)),
            )
            for i in input
        ]

    return create_futures_func


def run_func_threads(input, func=None, config=None, name=None, compute_id=None):
    from cubed.runtime.utils import execute_with_stats

    # TODO: can't use the execution_stats decorator since we get:
    # AttributeError: 'functools.partial' object has no attribute '__name__'
    result, stats = execute_with_stats(func, input, config=config)
    return result, stats
