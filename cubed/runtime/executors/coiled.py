from typing import Any, Mapping, Optional, Sequence

import coiled
from dask.distributed import as_completed
from networkx import MultiDiGraph

from cubed.core.array import Callback
from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execution_stats, handle_callbacks


def exec_stage_func(func, m, coiled_kwargs, **kwargs):
    # TODO would be good to give the dask tasks useful names

    # coiled_kwargs are tokenized by coiled.run, so each stage will reconnect to same cluster
    return coiled.run(**coiled_kwargs)(execution_stats(func)).submit(m, **kwargs)


class CoiledFunctionsDagExecutor(DagExecutor):
    """An execution engine that uses Coiled Functions."""

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        array_names: Optional[Sequence[str]] = None,
        resume: Optional[bool] = None,
        **coiled_kwargs: Mapping[str, Any],
    ) -> None:
        # Note this currently only builds the task graph for each stage once it gets to that stage in computation
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]
            futures = []
            for m in pipeline.mappable:
                future_func = exec_stage_func(
                    pipeline.function, m, coiled_kwargs, config=pipeline.config
                )
                futures.append(future_func)

            # gather the results of the coiled functions
            ac = as_completed(futures)
            if callbacks is not None:
                for future in ac:
                    result, stats = future.result()
                    if name is not None:
                        stats["array_name"] = name
                    handle_callbacks(callbacks, stats)
