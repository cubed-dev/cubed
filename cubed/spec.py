from typing import Optional, Union
from warnings import warn

from cubed.runtime.types import Executor
from cubed.utils import convert_to_bytes


class Spec:
    """Specification of resources available to run a computation."""

    def __init__(
        self,
        work_dir: Union[str, None] = None,
        max_mem: Union[int, None] = None,
        allowed_mem: Union[int, str, None] = None,
        reserved_mem: Union[int, str, None] = 0,
        executor: Union[Executor, None] = None,
        storage_options: Union[dict, None] = None,
    ):
        """
        Specify resources available to run a computation.

        Parameters
        ----------
        work_dir : str or None
            The directory path (specified as an fsspec URL) used for storing intermediate data.
        max_mem : int, optional
            **Deprecated**. The maximum memory available to a worker for data use for the computation, in bytes.
        allowed_mem : int or str, optional
            The total memory available to a worker for running a task, in bytes.

            If int it should be >=0. If str it should be of form <value><unit> where unit can be kB, MB, GB, TB etc.
            This includes any ``reserved_mem`` that has been set.
        reserved_mem : int or str, optional
            The memory reserved on a worker for non-data use when running a task, in bytes.

            If int it should be >=0. If str it should be of form <value><unit> where unit can be kB, MB, GB, TB etc.
        executor : Executor, optional
            The default executor for running computations.
        storage_options : dict, optional
            Storage options to be passed to fsspec.
        """

        if max_mem is not None:
            warn(
                "`max_mem` is deprecated, please use `allowed_mem` instead",
                DeprecationWarning,
                stacklevel=2,
            )

        self._work_dir = work_dir

        self._reserved_mem = convert_to_bytes(reserved_mem or 0)
        if allowed_mem is None:
            self._allowed_mem = (max_mem or 0) + self.reserved_mem
        else:
            self._allowed_mem = convert_to_bytes(allowed_mem)

        self._executor = executor
        self._storage_options = storage_options

    @property
    def work_dir(self) -> Optional[str]:
        """The directory path (specified as an fsspec URL) used for storing intermediate data."""
        return self._work_dir

    @property
    def allowed_mem(self) -> int:
        """
        The total memory available to a worker for running a task, in bytes.

        This includes any ``reserved_mem`` that has been set.
        """
        return self._allowed_mem

    @property
    def reserved_mem(self) -> int:
        """
        The memory reserved on a worker for non-data use when running a task, in bytes.

        See Also
        --------
        cubed.measure_reserved_mem
        """
        return self._reserved_mem

    @property
    def executor(self) -> Optional[Executor]:
        """The default executor for running computations."""
        return self._executor

    @property
    def storage_options(self) -> Optional[dict]:
        """Storage options to be passed to fsspec."""
        return self._storage_options

    def __repr__(self) -> str:
        return (
            f"cubed.Spec(work_dir={self._work_dir}, allowed_mem={self._allowed_mem}, "
            f"reserved_mem={self._reserved_mem}, executor={self._executor}, storage_options={self._storage_options})"
        )

    def __eq__(self, other):
        if isinstance(other, Spec):
            return (
                self.work_dir == other.work_dir
                and self.allowed_mem == other.allowed_mem
                and self.reserved_mem == other.reserved_mem
                and self.executor == other.executor
                and self.storage_options == other.storage_options
            )
        else:
            return False
