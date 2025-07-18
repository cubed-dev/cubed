from typing import Any, Dict, Literal, Tuple, Union

T_Shape = Tuple[int, ...]

T_DType = Any  # TODO: improve this

# Regular chunks are where all chunks must be the same size, whereas
# rectangular chunks may be different sizes, and contain the size of
# each chunk along each axis.
# Zarr currently supports only regular chunks, although rectangular chunks
# are proposed in https://zarr.dev/zeps/draft/ZEP0003.html.
# Dask supports rectangular chunks.
# The general T_Chunks type is for specifying chunks, it is generally
# converted to a T_RectangularChunks or T_RegularChunks internally.
T_RegularChunks = Tuple[int, ...]
T_RectangularChunks = Tuple[Tuple[int, ...], ...]
T_Chunks = Union[
    int, T_RegularChunks, T_RectangularChunks, Dict[Any, Any], Literal["auto"]
]

T_Store = Any  # TODO: improve this
