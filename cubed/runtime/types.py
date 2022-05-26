from typing import Union

from rechunker.types import PipelineExecutor

try:
    from cubed.runtime.executors.beam import BeamDagExecutor

    Executor = Union[PipelineExecutor, BeamDagExecutor]

except ImportError:  # pragma: no cover
    Executor = PipelineExecutor
