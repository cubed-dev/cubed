import pytest

from cubed.primitive.blockwise import ChunkKeys
from cubed.utils import normalize_chunks


def test_chunk_keys_iter():
    chunks = (3, 2)
    chunks_normal = normalize_chunks(chunks, shape=(4, 5))
    chunk_keys = ChunkKeys(chunks_normal)
    assert list(chunk_keys) == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]


@pytest.mark.parametrize(
    ("start", "stop"), [(0, None), (3, None), (3, 3), (3, 4), (3, 5), (3, 6), (5, None)]
)
def test_chunk_keys_range(start, stop):
    chunks = (3, 2)
    chunks_normal = normalize_chunks(chunks, shape=(4, 5))
    chunk_keys = ChunkKeys(chunks_normal)
    all_keys = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert list(chunk_keys.range(start, stop)) == all_keys[slice(start, stop)]
