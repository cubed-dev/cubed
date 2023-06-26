from dataclasses import dataclass
from operator import mul
from typing import Optional, TypeVar, Union
from warnings import warn

import numpy as np
from toolz import map, reduce

from cubed.runtime.pipeline import already_computed
from cubed.runtime.types import Executor
from cubed.storage.zarr import open_if_lazy_zarr_array
from cubed.utils import chunk_memory, convert_to_bytes
from cubed.vendor.dask.array.core import normalize_chunks

from .plan import arrays_to_plan

sym_counter = 0


def gensym(name="array"):
    """Generate a name with an incrementing counter"""
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


# type representing either a CoreArray or a public-facing Array
T_ChunkedArray = TypeVar("T_ChunkedArray", bound="CoreArray")


class CoreArray:
    """Chunked array backed by Zarr storage.

    This class has the basic array attributes (``dtype``, ``shape``, ``chunks``, etc).
    The other array methods and attributes are provided in a subclass.
    """

    def __init__(self, name, zarray, spec, plan):
        self.name = name
        self._zarray = zarray
        self._shape = zarray.shape
        self._dtype = zarray.dtype
        self._chunks = normalize_chunks(
            zarray.chunks, shape=self.shape, dtype=self.dtype
        )
        # if no spec is supplied, use a default with local temp dir,
        # and a modest amount of memory (200MB, of which 100MB is reserved)
        self.spec = spec or Spec(
            None, allowed_mem=200_000_000, reserved_mem=100_000_000
        )
        self.plan = plan

    @property
    def zarray_maybe_lazy(self):
        """The underlying Zarr array or LazyZarrArray. Use this during planning, before the computation has started."""
        return self._zarray

    @property
    def zarray(self):
        """The underlying Zarr array. May only be used during the computation once the array has been created."""
        return open_if_lazy_zarr_array(self._zarray)

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

    @property
    def nbytes(self) -> int:
        """Number of bytes in array"""
        return self.size * self.dtype.itemsize

    @property
    def itemsize(self) -> int:
        """Length of one array element in bytes"""
        return self.dtype.itemsize

    def _read_stored(self):
        # Only works if the array has been computed
        if self.size > 0:
            # read back from zarr
            return self.zarray[...]
        else:
            # this case fails for zarr, so just return an empty array of the correct shape
            return np.empty(self.shape, dtype=self.dtype)

    def compute(
        self,
        *,
        executor=None,
        callbacks=None,
        optimize_graph=True,
        resume=None,
        **kwargs,
    ):
        """Compute this array, and any arrays that it depends on."""
        result = compute(
            self,
            executor=executor,
            callbacks=callbacks,
            optimize_graph=optimize_graph,
            resume=resume,
            **kwargs,
        )
        if result:
            return result[0]

    def rechunk(self: T_ChunkedArray, chunks) -> T_ChunkedArray:
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

    def visualize(self, filename="cubed", format=None, optimize_graph=True):
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
        return visualize(
            self, filename=filename, format=format, optimize_graph=optimize_graph
        )

    def __getitem__(self: T_ChunkedArray, key, /) -> T_ChunkedArray:
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
        return f"cubed.core.CoreArray<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


class Spec:
    """Specification of resources available to run a computation."""

    def __init__(
        self,
        work_dir: Union[str, None] = None,
        max_mem: Union[int, None] = None,
        allowed_mem: Union[int, str, None] = None,
        reserved_mem: Union[int, str, None] = 0,
        executor: Union[Executor, None] = None,
        storage_options: Union[dict, None] = None,
    ):
        """
        Specify resources available to run a computation.

        Parameters
        ----------
        work_dir : str or None
            The directory path (specified as an fsspec URL) used for storing intermediate data.
        max_mem : int, optional
            **Deprecated**. The maximum memory available to a worker for data use for the computation, in bytes.
        allowed_mem : int or str, optional
            The total memory available to a worker for running a task, in bytes.

            If int it should be >=0. If str it should be of form <value><unit> where unit can be kB, MB, GB, TB etc.
            This includes any ``reserved_mem`` that has been set.
        reserved_mem : int or str, optional
            The memory reserved on a worker for non-data use when running a task, in bytes.

            If int it should be >=0. If str it should be of form <value><unit> where unit can be kB, MB, GB, TB etc.
        executor : Executor, optional
            The default executor for running computations.
        storage_options : dict, optional
            Storage options to be passed to fsspec.
        """

        if max_mem is not None:
            warn(
                "`max_mem` is deprecated, please use `allowed_mem` instead",
                DeprecationWarning,
                stacklevel=2,
            )

        self._work_dir = work_dir

        self._reserved_mem = convert_to_bytes(reserved_mem or 0)
        if allowed_mem is None:
            self._allowed_mem = (max_mem or 0) + self.reserved_mem
        else:
            self._allowed_mem = convert_to_bytes(allowed_mem)

        self._executor = executor
        self._storage_options = storage_options

    @property
    def work_dir(self) -> Optional[str]:
        """The directory path (specified as an fsspec URL) used for storing intermediate data."""
        return self._work_dir

    @property
    def allowed_mem(self) -> int:
        """
        The total memory available to a worker for running a task, in bytes.

        This includes any ``reserved_mem`` that has been set.
        """
        return self._allowed_mem

    @property
    def reserved_mem(self) -> int:
        """
        The memory reserved on a worker for non-data use when running a task, in bytes.

        See Also
        --------
        cubed.measure_reserved_mem
        """
        return self._reserved_mem

    @property
    def executor(self) -> Optional[Executor]:
        """The default executor for running computations."""
        return self._executor

    @property
    def storage_options(self) -> Optional[dict]:
        """Storage options to be passed to fsspec."""
        return self._storage_options

    def __repr__(self) -> str:
        return (
            f"cubed.Spec(work_dir={self._work_dir}, allowed_mem={self._allowed_mem}, "
            f"reserved_mem={self._reserved_mem}, executor={self._executor}, storage_options={self._storage_options})"
        )

    def __eq__(self, other):
        if isinstance(other, Spec):
            return (
                self.work_dir == other.work_dir
                and self.allowed_mem == other.allowed_mem
                and self.reserved_mem == other.reserved_mem
                and self.executor == other.executor
                and self.storage_options == other.storage_options
            )
        else:
            return False


class Callback:
    """Object to receive callback events during array computation."""

    def on_compute_start(self, dag, resume):
        """Called when the computation is about to start.

        Parameters
        ----------
        dag : networkx.MultiDiGraph
            The computation DAG.
        """
        pass  # pragma: no cover

    def on_compute_end(self, dag):
        """Called when the computation has finished.

        Parameters
        ----------
        dag : networkx.MultiDiGraph
            The computation DAG.
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
    """Callback information about a completed task (or tasks)."""

    array_name: str
    """Name of the array that the task is for."""

    num_tasks: int = 1
    """Number of tasks that this event applies to (default 1)."""

    task_create_tstamp: Optional[float] = None
    """Timestamp of when the task was created by the client."""

    function_start_tstamp: Optional[float] = None
    """Timestamp of when the function started executing on the remote worker."""

    function_end_tstamp: Optional[float] = None
    """Timestamp of when the function finished executing on the remote worker."""

    task_result_tstamp: Optional[float] = None
    """Timestamp of when the result of the task was received by the client."""

    peak_measured_mem_start: Optional[int] = None
    """Peak memory usage measured on the remote worker before the function starts executing."""

    peak_measured_mem_end: Optional[int] = None
    """Peak memory usage measured on the remote worker after the function finishes executing."""


def check_array_specs(arrays):
    specs = [a.spec for a in arrays if hasattr(a, "spec")]
    if not all(s == specs[0] for s in specs):
        raise ValueError(
            f"Arrays must have same spec in single computation. Specs: {specs}"
        )
    return arrays[0].spec


def compute(
    *arrays,
    executor=None,
    callbacks=None,
    optimize_graph=True,
    resume=None,
    **kwargs,
):
    """Compute multiple arrays at once."""
    plan = arrays_to_plan(*arrays)  # guarantees all arrays have same spec
    if executor is None:
        executor = arrays[0].spec.executor
        if executor is None:
            from cubed.runtime.executors.python import PythonDagExecutor

            executor = PythonDagExecutor()

    _return_in_memory_array = kwargs.pop("_return_in_memory_array", True)
    plan.execute(
        executor=executor,
        callbacks=callbacks,
        optimize_graph=optimize_graph,
        resume=resume,
        array_names=[a.name for a in arrays],
        **kwargs,
    )

    if _return_in_memory_array:
        return tuple(a._read_stored() for a in arrays)


def visualize(*arrays, filename="cubed", format=None, optimize_graph=True):
    """Produce a visualization of the computation graph for multiple arrays.

    Parameters
    ----------
    arrays : cubed.CoreArray
        The arrays to include in the visualization.
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
    plan = arrays_to_plan(*arrays)
    return plan.visualize(
        filename=filename, format=format, optimize_graph=optimize_graph
    )


class PeakMeasuredMemoryCallback(Callback):
    def on_task_end(self, event):
        self.peak_measured_mem = event.peak_measured_mem_end


def measure_reserved_memory(
    executor: Executor, work_dir: Optional[str] = None, **kwargs
) -> int:
    warn(
        "`measure_reserved_memory` is deprecated, please use `measure_reserved_mem` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return measure_reserved_mem(executor, work_dir=work_dir, **kwargs)


def measure_reserved_mem(
    executor: Executor, work_dir: Optional[str] = None, **kwargs
) -> int:
    """Measures the reserved memory use for a given executor runtime.

    This is the memory used by the Python process for running a task,
    excluding any memory used for data for the computation. It can vary by
    operating system, Python version, executor runtime, and installed package
    versions.

    It can be used as a guide to set ``reserved_mem`` when creating a ``Spec`` object.

    This implementation works by running a trivial computation on a tiny amount of
    data and measuring the peak memory use.

    Parameters
    ----------
    executor : cubed.runtime.types.Executor
        The executor to use to run the computation. It must be an executor that
        reports peak memory, such as Lithops or Modal.

    work_dir : str or None, optional
        The directory path (specified as an fsspec URL) used for storing intermediate data.
        This is required when using a cloud runtime.

    kwargs
        Keyword arguments to pass to the ``compute`` function.

    Returns
    -------
    int
        The memory used by the runtime in bytes.
    """
    import cubed.array_api as xp

    # give a generous memory allowance
    a = xp.ones((1,), spec=Spec(work_dir, allowed_mem="500MB"))
    b = xp.negative(a)
    peak_measured_mem_callback = PeakMeasuredMemoryCallback()
    b.compute(executor=executor, callbacks=[peak_measured_mem_callback], **kwargs)
    return peak_measured_mem_callback.peak_measured_mem
