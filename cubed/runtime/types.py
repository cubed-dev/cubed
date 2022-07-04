from typing import Union

from rechunker.types import PipelineExecutor


class DagExecutor:
    def execute_dag(self, dag, **kwargs):
        raise NotImplementedError  # pragma: no cover


Executor = Union[PipelineExecutor, DagExecutor]
