import logging
import sys
import time
from contextlib import contextmanager

from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback

REFRESH_PER_SECOND = 10


class RichProgressBar(Callback):
    """Rich progress bar for a computation."""

    def on_compute_start(self, event):
        # Set the pulse_style to the background colour to disable pulsing,
        # since Rich will pulse all non-started bars.
        logger_aware_progress = LoggerAwareProgress(
            SpinnerWhenRunningColumn(),
            TextColumn("[progress.description]{task.description}"),
            LeftJustifiedMofNCompleteColumn(),
            BarColumn(bar_width=None, pulse_style="bar.back"),
            TaskProgressColumn(
                text_format="[progress.percentage]{task.percentage:>3.1f}%"
            ),
            TimeElapsedColumn(),
            logger=logging.getLogger(),
        )
        progress = logger_aware_progress.__enter__()

        progress_tasks = {}
        for name, node in visit_nodes(event.dag, event.resume):
            num_tasks = node["primitive_op"].num_tasks
            op_display_name = node["op_display_name"].replace("\n", " ")
            progress_task = progress.add_task(
                f"{op_display_name}", start=False, total=num_tasks
            )
            progress_tasks[name] = progress_task

        self.logger_aware_progress = logger_aware_progress
        self.progress = progress
        self.progress_tasks = progress_tasks
        self.last_updated = time.time()

    def on_compute_end(self, event):
        self.logger_aware_progress.__exit__(None, None, None)

    def on_operation_start(self, event):
        self.progress.start_task(self.progress_tasks[event.name])

    def on_task_end(self, event):
        now = time.time()
        refresh = now - self.last_updated > (1.0 / REFRESH_PER_SECOND)
        self.last_updated = now
        self.progress.update(
            self.progress_tasks[event.name], advance=event.num_tasks, refresh=refresh
        )


class SpinnerWhenRunningColumn(SpinnerColumn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Override so spinner is not shown when bar has not yet started
    def render(self, task: "Task") -> RenderableType:
        text = (
            self.finished_text
            if not task.started or task.finished
            else self.spinner.render(task.get_time())
        )
        return text


class LeftJustifiedMofNCompleteColumn(MofNCompleteColumn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, task: "Task") -> Text:
        """Show completed/total."""
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return Text(
            f"{completed}{self.separator}{total}".ljust(total_width + 1 + total_width),
            style="progress.download",
        )


# Based on CustomProgress from https://github.com/Textualize/rich/discussions/1578
@contextmanager
def LoggerAwareProgress(*args, **kwargs):
    """Wrapper around rich.progress.Progress to manage logging output to stderr."""
    try:
        __logger = kwargs.pop("logger", None)
        streamhandlers = [
            x for x in __logger.root.handlers if type(x) is logging.StreamHandler
        ]

        with Progress(*args, **kwargs) as progress:
            for handler in streamhandlers:
                __prior_stderr = handler.stream
                handler.setStream(sys.stderr)

            yield progress

    finally:
        streamhandlers = [
            x for x in __logger.root.handlers if type(x) is logging.StreamHandler
        ]
        for handler in streamhandlers:
            handler.setStream(__prior_stderr)
