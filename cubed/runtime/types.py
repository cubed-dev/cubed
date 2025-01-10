from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, ClassVar, Iterable, Optional

from networkx import MultiDiGraph

from cubed.vendor.rechunker.types import Config


class DagExecutor:
    @property
    def name(self) -> str:
        raise NotImplementedError  # pragma: no cover

    def execute_dag(self, dag: MultiDiGraph, **kwargs) -> None:
        raise NotImplementedError  # pragma: no cover


Executor = DagExecutor


@dataclass(frozen=True)
class CubedPipeline:
    """Generalisation of rechunker ``Pipeline`` with extra attributes."""

    function: Callable[..., Any]
    name: str
    mappable: Iterable
    config: Config


@dataclass
class ComputeStartEvent:
    """Callback information about a computation that is about to start."""

    compute_id: str
    """ID of the computation."""

    dag: MultiDiGraph
    """The computation DAG."""

    resume: bool
    """If the computation has been resumed."""


@dataclass
class ComputeEndEvent:
    """Callback information about a computation that has finished."""

    compute_id: str
    """ID of the computation."""

    dag: MultiDiGraph
    """The computation DAG."""


@dataclass
class OperationStartEvent:
    """Callback information about an operation that is about to start."""

    name: str
    """Name of the operation."""


@dataclass
class TaskEndEvent:
    """Callback information about a completed task (or tasks)."""

    name: str
    """Name of the operation that the task is for."""

    num_tasks: int = 1
    """Number of tasks that this event applies to (default 1)."""

    result: Optional[Any] = None
    """Return value of the task."""

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


class Callback:
    """Object to receive callback events during array computation."""

    active: ClassVar[set["Callback"]] = set()

    def __enter__(self):
        self.register()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.unregister()

    def register(self) -> None:
        Callback.active.add(self)

    def unregister(self) -> None:
        Callback.active.remove(self)

    def on_compute_start(self, event):
        """Called when the computation is about to start.

        Parameters
        ----------
        event : ComputeStartEvent
            Information about the computation.
        """
        pass  # pragma: no cover

    def on_compute_end(self, ComputeEndEvent):
        """Called when the computation has finished.

        Parameters
        ----------
        event : ComputeStartEvent
            Information about the computation.
        """
        pass  # pragma: no cover

    def on_operation_start(self, event):
        pass

    def on_task_end(self, event):
        """Called when the a task ends.

        Parameters
        ----------
        event : TaskEndEvent
            Information about the task execution.
        """
        pass  # pragma: no cover
