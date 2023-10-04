from dataclasses import dataclass
from typing import Optional

from networkx import MultiDiGraph


class DagExecutor:
    def execute_dag(self, dag: MultiDiGraph, **kwargs) -> None:
        raise NotImplementedError  # pragma: no cover


Executor = DagExecutor


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
