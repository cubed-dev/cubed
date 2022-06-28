import contextlib
import sys
from operator import mul

import networkx as nx
import numpy as np
from dask.array.core import normalize_chunks
from toolz import map, reduce

from cubed.runtime.pipeline import already_computed

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
    def device(self):
        return "cpu"

    @property
    def mT(self):
        from cubed.array_api.linear_algebra_functions import matrix_transpose

        return matrix_transpose(self)

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

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError("x.T requires x to have 2 dimensions.")
        from cubed.array_api.linear_algebra_functions import matrix_transpose

        return matrix_transpose(self)

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

    def __getitem__(self, key, /):
        from cubed.core.ops import index

        return index(self, key)

    def __setitem__(self, key, value):
        if isinstance(value, Array) and value.ndim != 0:
            raise NotImplementedError(
                "Calling __setitem__ on an array with more than 0 dimensions is not supported."
            )

        nodes = {n: d for (n, d) in self.plan.dag.nodes(data=True)}
        if not already_computed(nodes[self.name]):
            raise NotImplementedError(
                "Calling __setitem__ on an array that has not been computed is not supported."
            )

        self.zarray.__setitem__(key, value)

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

    def __eq__(self, other, /):
        from cubed.core.ops import elemwise

        return elemwise(np.equal, self, other, dtype=np.bool_)

    def __neg__(self, /):
        from cubed.core.ops import elemwise

        return elemwise(np.negative, self)

    def __repr__(self):
        return f"Array<{self.name}, shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


class Callback:
    def on_compute_start(self, arr):
        pass

    def on_compute_end(self, arr):
        pass

    def on_task_end(self, name=None):
        pass


class TqdmProgressBar(Callback):
    """Progress bar for a computation."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def on_compute_start(self, arr):
        from tqdm import tqdm

        self.pbars = {}
        i = 0
        dag = arr.plan.optimize(arr.name)
        nodes = {n: d for (n, d) in dag.nodes(data=True)}
        for node in list(nx.topological_sort(dag)):
            if already_computed(nodes[node]):
                continue
            num_tasks = nodes[node]["num_tasks"]
            self.pbars[node] = tqdm(
                *self.args, desc=node, total=num_tasks, position=i, **self.kwargs
            )
            i = i + 1

    def on_compute_end(self, arr):
        for pbar in self.pbars.values():
            pbar.close()

    def on_task_end(self, name=None):
        self.pbars[name].update()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """
    Context manager for redirecting stdout and stderr when using tqdm.
    See https://github.com/tqdm/tqdm#redirecting-writing
    """
    from tqdm.contrib import DummyTqdmFile

    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
