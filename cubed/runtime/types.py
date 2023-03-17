from typing import Union

from cubed.vendor.rechunker.types import PipelineExecutor


class DagExecutor:
    def execute_dag(self, dag, **kwargs):
        raise NotImplementedError  # pragma: no cover


Executor = Union[PipelineExecutor, DagExecutor]
