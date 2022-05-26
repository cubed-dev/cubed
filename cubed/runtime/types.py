from typing import Union

from rechunker.types import PipelineExecutor


class DagExecutor:
    @staticmethod
    def execute_dag(dag, **kwargs):
        raise NotImplementedError


Executor = Union[PipelineExecutor, DagExecutor]
