from dataclasses import dataclass
from operator import mul
from typing import Optional

import numpy as np
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

    This class has the basic array attributes (``dtype``, ``shape``, ``chunks``, etc).
    The other array methods and attributes are provided in a subclass.
    """

    def __init__(self, name, zarray, plan):
        self.name = name
        self.zarray = zarray
        self._shape = zarray.shape
        self._dtype = zarray.dtype
        self._chunks = normalize_chunks(
            zarray.chunks, shape=self.shape, dtype=self.dtype
        )
        self.plan = plan

    @classmethod
    def _new(cls, name, zarray, plan):
        # Always create an Array object subclass that has array API methods and attributes
        from cubed.array_api.array_object import Array

        return Array(name, zarray, plan)

    @property
    def chunkmem(self):
        """Amount of memory in bytes that a single chunk uses."""
        return chunk_memory(self.dtype, self.chunksize)

    @property
    def chunksize(self):
        """A tuple indicating the chunk size of each corresponding array dimension."""
        return tuple(max(c) for c in self.chunks)

    @property
    def chunks(self):
        """A tuple containing a sequence of block sizes for each corresponding array dimension."""
        return self._chunks

    @property
    def dtype(self):
        """Data type of the array elements."""
        return self._dtype

    @property
    def ndim(self):
        """Number of array dimensions (axes)."""
        return len(self.shape)

    @property
    def numblocks(self):
        """A tuple indicating the number of blocks (chunks) in each corresponding array dimension."""
        return tuple(map(len, self.chunks))

    @property
    def npartitions(self):
        """Number of chunks in the array."""
        return reduce(mul, self.numblocks, 1)

    @property
    def shape(self):
        """Array dimensions."""
        return self._shape

    @property
    def size(self):
        """Number of elements in the array."""
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
        """Compute this array, and any arrays that it depends on."""
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
            if self.size > 0:
                # read back from zarr
                return self.zarray[...]
            else:
                # this case fails for zarr, so just return an empty array of the correct shape
                return np.empty(self.shape, dtype=self.dtype)

    def rechunk(self, chunks):
        """Change the chunking of this array without changing its shape or data.

        Parameters
        ----------
        chunks : tuple
            The desired chunks of the array after rechunking.

        Returns
        -------
        cubed.CoreArray
            An array with the desired chunks.
        """
        from cubed.core.ops import rechunk

        return rechunk(self, chunks)

    def visualize(self, *, filename="cubed", format=None, optimize_graph=True):
        """Produce a visualization of the computation graph for this array.

        Parameters
        ----------
        filename : str
            The name of the file to write to disk. If the provided ``filename``
            doesn't include an extension, '.svg' will be used by default.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file.  Default is 'svg'.
        optimize_graph : bool, optional
            If True, the graph is optimized before rendering.  Otherwise,
            the graph is displayed as is. Default is True.

        Returns
        -------
        IPython.display.SVG, or None
            An IPython SVG image if IPython can be imported (for rendering
            in a notebook), otherwise None.
        """
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
        """Called when the computation is about to start.

        Parameters
        ----------
        arr : cubed.CoreArray
            The array being computed.
        """
        pass  # pragma: no cover

    def on_compute_end(self, arr):
        """Called when the computation has finished.

        Parameters
        ----------
        arr : cubed.CoreArray
            The array being computed.
        """
        pass  # pragma: no cover

    def on_task_end(self, event):
        """Called when the a task ends.

        Parameters
        ----------
        event : TaskEndEvent
            Information about the task execution.
        """
        pass  # pragma: no cover


@dataclass
class TaskEndEvent:
    """Callback information about a completed task."""

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
