from typing import Union

from rechunker.types import PipelineExecutor

try:
    from barry.rechunker_extensions.executors.beam import BeamDagExecutor

    Executor = Union[PipelineExecutor, BeamDagExecutor]

except ImportError:
    Executor = PipelineExecutor
