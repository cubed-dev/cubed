from networkx import MultiDiGraph


class DagExecutor:
    def execute_dag(self, dag: MultiDiGraph, **kwargs) -> None:
        raise NotImplementedError  # pragma: no cover


Executor = DagExecutor
