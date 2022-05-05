# TODO: port to rechunker (use ChunkKeys Iterable instead of a generator, to avoid beam pickle error)
from typing import Iterable, Tuple

from rechunker.pipeline import (
    chunk_keys,
    copy_intermediate_to_write,
    copy_read_to_intermediate,
    copy_read_to_write,
)
from rechunker.types import CopySpec, Pipeline, Stage

from .utils import gensym


class ChunkKeys(Iterable[Tuple[slice, ...]]):
    def __init__(self, shape: Tuple[int, ...], chunks: Tuple[int, ...]):
        self.shape = shape
        self.chunks = chunks

    def __iter__(self):
        return chunk_keys(self.shape, self.chunks)


def spec_to_pipeline(spec: CopySpec) -> Pipeline:
    # typing won't work until we start using numpy types
    shape = spec.read.array.shape  # type: ignore
    if spec.intermediate.array is None:
        stages = [
            Stage(
                copy_read_to_write,
                gensym("copy_read_to_write"),
                mappable=ChunkKeys(shape, spec.write.chunks),
            )
        ]
    else:
        stages = [
            Stage(
                copy_read_to_intermediate,
                gensym("copy_read_to_intermediate"),
                mappable=ChunkKeys(shape, spec.intermediate.chunks),
            ),
            Stage(
                copy_intermediate_to_write,
                gensym("copy_intermediate_to_write"),
                mappable=ChunkKeys(shape, spec.write.chunks),
            ),
        ]
    return Pipeline(stages, config=spec)
