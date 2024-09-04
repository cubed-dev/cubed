import contextlib
import sys

from toolz import map

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback


class TqdmProgressBar(Callback):
    """Progress bar for a computation."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def on_compute_start(self, event):
        from tqdm.auto import tqdm

        # find the maximum display width so we can align bars below
        max_op_display_name = (
            max(
                len(node["op_display_name"].replace("\n", " "))
                for _, node in visit_nodes(event.dag, event.resume)
            )
            + 1  # for the colon
        )

        self.pbars = {}
        for i, (name, node) in enumerate(visit_nodes(event.dag, event.resume)):
            num_tasks = node["primitive_op"].num_tasks
            op_display_name = node["op_display_name"].replace("\n", " ") + ":"
            # note double curlies to get literal { and } for tqdm bar format
            bar_format = (
                f"{{desc:{max_op_display_name}}} {{percentage:3.0f}}%|{{bar}}{{r_bar}}"
            )
            self.pbars[name] = tqdm(
                *self.args,
                desc=op_display_name,
                total=num_tasks,
                position=i,
                bar_format=bar_format,
                **self.kwargs,
            )

    def on_compute_end(self, event):
        for pbar in self.pbars.values():
            pbar.close()

    def on_task_end(self, event):
        self.pbars[event.name].update(event.num_tasks)


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
