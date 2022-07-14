import contextlib
import sys

import networkx as nx
from toolz import map

from cubed.core.array import Callback
from cubed.runtime.pipeline import already_computed


class TqdmProgressBar(Callback):
    """Progress bar for a computation."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def on_compute_start(self, arr):
        from tqdm.auto import tqdm

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

    def on_task_end(self, event):
        self.pbars[event.array_name].update()


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
