from dataclasses import dataclass
from operator import mul
from typing import Optional

from dask.array.core import normalize_chunks
from toolz import map, reduce

from cubed.runtime.pipeline import already_computed
from cubed.utils import chunk_memory

sym_counter = 0


def gensym(name="array"):
    """Generate a name with an incrementing counter"""
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


class CoreArray:
    """Chunked array backed by Zarr storage.

    This class has the basic array attributes (`dtype`, `shape`, `chunks`, etc).
    The other array methods and attributes are provided in a subclass.
    """

    def __init__(self, name, zarray, plan):
        self.name = name
        self.zarray = zarray
        self.shape = zarray.shape
        self.dtype = zarray.dtype
        self.chunks = normalize_chunks(
            zarray.chunks, shape=self.shape, dtype=self.dtype
        )
        self.plan = plan

    @classmethod
    def new(cls, name, zarray, plan):
        # Always create an Array object subclass that has array API methods and attributes
        from cubed.array_api.array_object import Array

        return Array(name, zarray, plan)

    @property
    def chunkmem(self):
        return chunk_memory(self.dtype, self.chunksize)

    @property
    def chunksize(self):
        return tuple(max(c) for c in self.chunks)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numblocks(self):
        return tuple(map(len, self.chunks))

    @property
    def npartitions(self):
        return reduce(mul, self.numblocks, 1)

    @property
    def size(self):
        return reduce(mul, self.shape, 1)

    def compute(
        self,
        *,
        return_stored=True,
        executor=None,
        callbacks=None,
        optimize_graph=True,
        **kwargs,
    ):
        if callbacks is not None:
            [callback.on_compute_start(self) for callback in callbacks]

        self.plan.execute(
            self.name,
            executor=executor,
            callbacks=callbacks,
            optimize_graph=optimize_graph,
            **kwargs,
        )

        if callbacks is not None:
            [callback.on_compute_end(self) for callback in callbacks]

        if return_stored:
            # read back from zarr
            return self.zarray[...]

    def visualize(self, filename="cubed", format=None, optimize_graph=True):
        return self.plan.visualize(
            filename=filename, format=format, optimize_graph=optimize_graph
        )

    def __getitem__(self, key, /):
        from cubed.core.ops import index

        return index(self, key)

    def __setitem__(self, key, value):
        if isinstance(value, CoreArray) and value.ndim != 0:
            raise NotImplementedError(
                "Calling __setitem__ on an array with more than 0 dimensions is not supported."
            )

        nodes = {n: d for (n, d) in self.plan.dag.nodes(data=True)}
        if not already_computed(nodes[self.name]):
            raise NotImplementedError(
                "Calling __setitem__ on an array that has not been computed is not supported."
            )

        self.zarray.__setitem__(key, value)

    def __repr__(self):
        return f"CoreArray<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


class Callback:
    """Object to receive callback events during array computation."""

    def on_compute_start(self, arr):
        pass  # pragma: no cover

    def on_compute_end(self, arr):
        pass  # pragma: no cover

    def on_task_end(self, event):
        pass  # pragma: no cover


@dataclass
class TaskEndEvent:
    """Information about a completed task."""

    array_name: str
    """Name of the array that the task is for."""

    task_create_tstamp: Optional[float] = None
    """Timestamp of when the task was created by the client."""

    function_start_tstamp: Optional[float] = None
    """Timestamp of when the function started executing on the remote worker."""

    function_end_tstamp: Optional[float] = None
    """Timestamp of when the function finished executing on the remote worker."""

    task_result_tstamp: Optional[float] = None
    """Timestamp of when the result of the task was received by the client."""

    peak_memory_start: Optional[int] = None
    """Peak memory usage on the remote worker before the function starts executing."""

    peak_memory_end: Optional[int] = None
    """Peak memory usage on the remote worker after the function finishes executing."""
