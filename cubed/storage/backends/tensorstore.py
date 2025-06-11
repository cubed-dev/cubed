import dataclasses
import math
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorstore

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store
from cubed.utils import join_path


@dataclasses.dataclass(frozen=True)
class TensorStoreArray:
    array: tensorstore.TensorStore

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype.numpy_dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        return self.array.chunk_layout.read_chunk.shape or ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def oindex(self):
        return self.array.oindex

    def __getitem__(self, key):
        # read eagerly
        return self.array.__getitem__(key).read().result()

    def __setitem__(self, key, value):
        self.array.__setitem__(key, value)


class TensorStoreGroup(dict):
    def __init__(
        self,
        shape: Optional[T_Shape] = None,
        dtype: Optional[T_DType] = None,
        chunks: Optional[T_RegularChunks] = None,
    ):
        dict.__init__(self)
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        return {field: zarray[key] for field, zarray in self.items()}

    def set_basic_selection(self, selection, value, fields=None):
        self[fields][selection] = value


def encode_dtype(d):
    if d.fields is None:
        return d.str
    else:
        return d.descr


def get_metadata(dtype, chunks, compressor):
    metadata = {}
    if dtype is not None:
        dtype = np.dtype(dtype)
        metadata["dtype"] = encode_dtype(dtype)
    if chunks is not None:
        if isinstance(chunks, int):
            chunks = (chunks,)
        metadata["chunks"] = chunks
    if compressor != "default":
        metadata["compressor"] = compressor
    return metadata


def open_tensorstore_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    compressor: Union[dict, str, None] = "default",
    **kwargs,
):
    store = str(store)  # TODO: check if Path or str

    spec: Dict[str, Any]
    if "://" in store:
        spec = {"driver": "zarr", "kvstore": store}
    else:
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": store},
            "path": path or "",
        }

    if mode == "r":
        open_kwargs = dict(read=True, open=True)
    elif mode == "r+":
        open_kwargs = dict(read=True, write=True, open=True)
    elif mode == "a":
        open_kwargs = dict(read=True, write=True, open=True, create=True)
    elif mode == "w":
        open_kwargs = dict(read=True, write=True, create=True, delete_existing=True)
    elif mode == "w-":
        open_kwargs = dict(read=True, write=True, create=True)
    else:
        raise ValueError(f"Mode not supported: {mode}")

    if dtype is None or not hasattr(dtype, "fields") or dtype.fields is None:
        metadata = get_metadata(dtype, chunks, compressor)
        if metadata:
            spec["metadata"] = metadata

        return TensorStoreArray(
            tensorstore.open(
                spec,
                shape=shape,
                dtype=dtype,
                **open_kwargs,
            ).result()
        )
    else:
        ret = TensorStoreGroup(shape=shape, dtype=dtype, chunks=chunks)
        for field in dtype.fields:
            field_path = field if path is None else join_path(path, field)
            spec["path"] = field_path

            field_dtype, _ = dtype.fields[field]
            metadata = get_metadata(field_dtype, chunks, compressor)
            if metadata:
                spec["metadata"] = metadata

            ret[field] = TensorStoreArray(
                tensorstore.open(
                    spec,
                    shape=shape,
                    dtype=field_dtype,
                    **open_kwargs,
                ).result()
            )
        return ret
