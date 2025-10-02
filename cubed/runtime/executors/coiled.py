from typing import Any, Mapping, Optional, Sequence

import coiled
from networkx import MultiDiGraph

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import (
    execution_stats,
    handle_callbacks,
    handle_operation_start_callbacks,
)
from cubed.spec import Spec


def make_coiled_function(func, name, coiled_kwargs):
    return coiled.function(**coiled_kwargs)(execution_stats(func, name=name))


class CoiledExecutor(DagExecutor):
    """An execution engine that uses Coiled Functions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "coiled"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **coiled_kwargs: Mapping[str, Any],
    ) -> None:
        merged_kwargs = {**self.kwargs, **coiled_kwargs}
        minimum_workers = merged_kwargs.pop("minimum_workers", None)
        # Note this currently only builds the task graph for each stage once it gets to that stage in computation
        for name, node in visit_nodes(dag):
            handle_operation_start_callbacks(callbacks, name)
            pipeline = node["pipeline"]
            # this name will show up on the dask dashboard - need to replace '-' as anything after it is suppressed
            op_display_name = (
                node["op_display_name"].replace("\n", " ").replace("-", "_")
            )
            coiled_function = make_coiled_function(
                pipeline.function, op_display_name, merged_kwargs
            )
            if minimum_workers is not None:
                coiled_function.cluster.adapt(minimum=minimum_workers)
            # coiled expects a sequence (it calls `len` on it)
            input = list(pipeline.mappable)
            for result, stats in coiled_function.map(input, config=pipeline.config):
                if callbacks is not None:
                    if name is not None:
                        stats["name"] = name
                    handle_callbacks(callbacks, result, stats)
