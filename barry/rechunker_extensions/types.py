from typing import Union

from rechunker.types import PipelineExecutor

from barry.rechunker_extensions.executors.beam import BeamDagExecutor

Executor = Union[PipelineExecutor, BeamDagExecutor]
