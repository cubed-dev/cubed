import pathlib

import anywidget
import traitlets
from IPython.display import display

from cubed.runtime.types import Callback


class TimelineWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "timeline_widget.js"

    width = traitlets.Unicode("100%").tag(sync=True)
    height = traitlets.Unicode("300px").tag(sync=True)


class LiveTimelineViewer(Callback):
    def __init__(self, widget=None, *, width="100%", height="300px"):
        self.widget = widget
        self.width = width
        self.height = height
        self.task_index = 0
        self.t0 = 0

    def on_compute_start(self, event):
        if self.widget is None:
            self.widget = TimelineWidget(
                width=self.width,
                height=self.height,
            )
            # show the widget in the current cell
            display(self.widget)

        self.widget.send({"type": "on_compute_start"})

    def on_task_end(self, event):
        if self.t0 == 0:
            self.t0 = event.task_create_tstamp
        self.widget.send(
            {
                "type": "on_task_end",
                "values": [
                    {
                        "name": event.name,
                        "task_index": self.task_index,
                        "elapsed_time": event.task_create_tstamp - self.t0,
                        "event_type": "task create",
                    },
                    {
                        "name": event.name,
                        "task_index": self.task_index,
                        "elapsed_time": event.function_start_tstamp - self.t0,
                        "event_type": "function start",
                    },
                    {
                        "name": event.name,
                        "task_index": self.task_index,
                        "elapsed_time": event.function_end_tstamp - self.t0,
                        "event_type": "function end",
                    },
                    {
                        "name": event.name,
                        "task_index": self.task_index,
                        "elapsed_time": event.task_result_tstamp - self.t0,
                        "event_type": "task result",
                    },
                ],
            }
        )

        self.task_index += 1
