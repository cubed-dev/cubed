from typing import Any, Mapping, Optional, Sequence

import coiled
from networkx import MultiDiGraph

from cubed.core.array import Callback, Spec
from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor
from cubed.runtime.utils import execution_stats, handle_callbacks


def make_coiled_function(func, coiled_kwargs):
    return coiled.function(**coiled_kwargs)(execution_stats(func))


class CoiledFunctionsDagExecutor(DagExecutor):
    """An execution engine that uses Coiled Functions."""

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        array_names: Optional[Sequence[str]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        **coiled_kwargs: Mapping[str, Any],
    ) -> None:
        # Note this currently only builds the task graph for each stage once it gets to that stage in computation
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]
            coiled_function = make_coiled_function(pipeline.function, coiled_kwargs)
            input = list(
                pipeline.mappable
            )  # coiled expects a sequence (it calls `len` on it)
            for _, stats in coiled_function.map(input, config=pipeline.config):
                if callbacks is not None:
                    if name is not None:
                        stats["array_name"] = name
                    handle_callbacks(callbacks, stats)
