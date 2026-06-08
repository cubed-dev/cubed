from collections.abc import Sequence

from networkx import MultiDiGraph

from cubed.runtime.types import Callback, DagExecutor
from cubed.spec import Spec


class RaiseIfComputesExecutor(DagExecutor):
    """An executor that always raises a runtime error.

    Useful for testing that a computation is not triggered eagerly.
    """

    @property
    def name(self) -> str:
        return "raise-if-computes"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Sequence[Callback] | None = None,
        spec: Spec | None = None,
        compute_id: str | None = None,
        **kwargs,
    ) -> None:
        raise RuntimeError("'compute' was called")
