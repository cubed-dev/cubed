"""Types definitions used by executors."""
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from mypy_extensions import NamedArg

# TODO: replace with Protocols, once Python 3.8+ is required
Array = Any
ReadableArray = Any
WriteableArray = Any


class ArrayProxy(NamedTuple):
    """Representation of a chunked array for reads and writes.

    Attributes
    ----------
    array : array or None
        If ``array`` is None, it represents an array without any on-disk
        representation. Otherwise ``array`` should be an explicit array to read
        and/or write to.
    chunks : tuple
        Chunks to use when reading/writing from this array. If ``array`` is
        chunked, ``array.chunks`` will always even divide ``chunks``.
    """

    array: Optional[Array]
    chunks: Tuple[int, ...]


class CopySpec(NamedTuple):
    """Specification for how to rechunk an array using a single intermediate.

    Attributes
    ----------
    read : ArrayProxy
        Read proxy with an ``array`` attribute that supports ``__getitem__``.
    intermediate : ArrayProxy
        Intermediate proxy with either an ``array`` that is either ``None``
        (no intermediate storage on disk) or that supports both ``__getitem__``
        and ``__setitem__``. The ``chunks`` on intermediates are technically
        redundant (they the elementwise minimum of the read and write chunks)
        but they are provided for convenience.
    write : ArrayProxy
        Write proxy with an ``array`` attribute that supports ``__setitem__``.
    """

    read: ArrayProxy
    intermediate: ArrayProxy
    write: ArrayProxy


Config = Any  # TODO: better typing for config
SingleArgumentStageFunction = Callable[
    [Any, NamedArg(type=Any, name="config")], None
]  # noqa: F821
NoArgumentStageFunction = Callable[
    [NamedArg(type=Any, name="config")], None
]  # noqa: F821
StageFunction = Union[NoArgumentStageFunction, SingleArgumentStageFunction]


@dataclass(frozen=True)
class Stage:
    function: StageFunction
    name: str
    mappable: Optional[Iterable] = None


@dataclass(frozen=True)
class Pipeline:
    stages: Iterable[Stage]
    config: Optional[Config] = None


# ParallelPipelines contains one or more MultiStagePipelines, to be executed in parallel
ParallelPipelines = Tuple[Pipeline, ...]

T = TypeVar("T")


class PipelineExecutor(Generic[T]):
    """Base class for pipeline-based execution engines.

    Executors prepare and execute scheduling plans, in whatever form is most
    convenient for users of that executor.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> T:
        """Convert pipeline specifications into a plan."""
        raise NotImplementedError

    def execute_plan(self, plan: T, **kwargs):
        """Execute a plan."""
        raise NotImplementedError


class CopySpecExecutor(Generic[T]):
    """Base class for copy-spec execution engines.

    Executors prepare and execute scheduling plans, in whatever form is most
    convenient for users of that executor.
    """

    def prepare_plan(self, specs: Iterable[CopySpec]) -> T:
        """Convert copy specifications into a plan."""
        raise NotImplementedError

    def execute_plan(self, plan: T, **kwargs):
        """Execute a plan."""
        raise NotImplementedError
