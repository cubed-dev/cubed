from operator import mul

import numpy as np
from dask.array.core import normalize_chunks
from toolz import map, reduce

sym_counter = 0


def gensym(name="array"):
    """Generate a name with an incrementing counter"""
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


class Array:
    """Chunked array backed by Zarr storage."""

    def __init__(self, name, zarray, plan):
        self.name = name
        self.zarray = zarray
        self.shape = zarray.shape
        self.dtype = zarray.dtype
        self.chunks = normalize_chunks(
            zarray.chunks, shape=self.shape, dtype=self.dtype
        )
        self.plan = plan

    def __array__(self, dtype=None):
        x = self.compute()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

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

    def visualize(self, filename="cubed", format=None):
        return self.plan.visualize(filename=filename, format=format)

    def __bool__(self, /):
        if self.ndim != 0:
            raise TypeError("bool is only allowed on arrays with 0 dimensions")
        return bool(self.compute())

    def __float__(self, /):
        if self.ndim != 0:
            raise TypeError("float is only allowed on arrays with 0 dimensions")
        return float(self.compute())

    def __int__(self, /):
        if self.ndim != 0:
            raise TypeError("int is only allowed on arrays with 0 dimensions")
        return int(self.compute())

    def __repr__(self):
        return f"Array<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


class Callback:
    def on_compute_start(self, arr):
        pass

    def on_compute_end(self, arr):
        pass

    def on_task_end(self, n=1):
        pass


class TqdmProgressBar(Callback):
    """Progress bar for a computation."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def on_compute_start(self, arr):
        from tqdm import tqdm

        total_tasks = arr.plan.num_tasks(arr.name)
        self.pbar = tqdm(*self.args, total=total_tasks, **self.kwargs)

    def on_compute_end(self, arr):
        self.pbar.close()

    def on_task_end(self, n=1):
        self.pbar.update(n)
