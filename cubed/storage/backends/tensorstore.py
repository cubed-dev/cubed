import dataclasses
import math
from typing import Optional

import numpy as np
import tensorstore

from cubed.types import T_DType, T_RegularChunks, T_Shape, T_Store


@dataclasses.dataclass(frozen=True)
class TensorstoreArray:
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


class TensorstoreStructuredArrays(dict):
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
        # create a numpy structured array from the tensorstore fields
        arrays = {field: ts_array[key] for field, ts_array in self.items()}
        array0 = next(iter(arrays.values()))  # TODO: not safe
        ret = np.empty(array0.shape, dtype=self.dtype)
        for field, arr in arrays.items():
            ret[field] = arr
        return ret

    def set_basic_selection(self, selection, value, fields=None):
        # TODO: multiple fields? disallow
        self[fields][selection] = value


def encode_dtype(d):
    if d.fields is None:
        return d.str
    else:
        return d.descr


def open_tensorstore_array(
    store: T_Store,
    mode: str,
    *,
    shape: Optional[T_Shape] = None,
    dtype: Optional[T_DType] = None,
    chunks: Optional[T_RegularChunks] = None,
    path: Optional[str] = None,
    **kwargs,
):
    store = str(store)  # TODO: check if Path or str

    if "://" in store:
        spec = {"driver": "zarr", "kvstore": store}
    else:
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": store},
            "path": path or "",
        }

    metadata = {}
    if dtype is not None:
        metadata["dtype"] = encode_dtype(dtype)
    if chunks is not None:
        metadata["chunks"] = chunks
    if metadata:
        spec["metadata"] = metadata

    if mode == "r":
        open_kwargs = dict(read=True, open=True)
    elif mode == "r+":
        open_kwargs = dict(read=True, write=True, open=True)
    elif mode == "a":
        open_kwargs = dict(read=True, write=True, open=True, create=True)
    elif mode == "w":
        open_kwargs = dict(write=True, create=True, delete_existing=True)
    elif mode == "w-":
        open_kwargs = dict(write=True, create=True)
    else:
        raise ValueError(f"Mode not supported: {mode}")

    if dtype is None or dtype.fields is None:
        return TensorstoreArray(
            tensorstore.open(
                spec,
                shape=shape,
                dtype=dtype,
                **open_kwargs,
            ).result()
        )
    else:
        ret = TensorstoreStructuredArrays(shape=shape, dtype=dtype, chunks=chunks)
        for field in dtype.fields:
            field_dtype, _ = dtype.fields[field]
            spec["field"] = field
            target = TensorstoreArray(
                tensorstore.open(
                    spec,
                    shape=shape,
                    dtype=field_dtype,
                    **open_kwargs,
                ).result()
            )
            ret[field] = target
        return ret
