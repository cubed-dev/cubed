from typing import List, Optional, Tuple

from cubed.types import T_Chunks, T_DType, T_RectangularChunks, T_Shape

def _check_regular_chunks(chunkset: T_RectangularChunks) -> bool: ...

def broadcast_chunks(
    *chunkss
) -> Tuple[Tuple[int, ...]]: ...

def common_blockdim(blockdims: List[Tuple[int, ...]]) -> Tuple[int, ...]: ...

def normalize_chunks(
    chunks: T_Chunks,
    shape: Optional[T_Shape] = None,
    limit: Optional[int] = None,
    dtype: Optional[T_DType] = None,
    previous_chunks: Optional[T_RectangularChunks] = None,
) -> T_RectangularChunks:
    ...
