from operator import mul
from typing import Optional, TypeVar
from warnings import warn

import numpy as np
from toolz import map, reduce

from cubed.runtime.types import Callback, Executor
from cubed.spec import Spec
from cubed.storage.zarr import open_if_lazy_zarr_array
from cubed.utils import chunk_memory
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

    def __repr__(self):
        return f"cubed.core.CoreArray<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


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
    spec = check_array_specs(arrays)  # guarantees all arrays have same spec
    plan = arrays_to_plan(*arrays)
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
        spec=spec,
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
