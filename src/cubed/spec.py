from functools import cached_property, lru_cache
from typing import Optional, Union

import donfig
from donfig.config_obj import expand_environment_variables

from cubed.runtime.create import create_executor
from cubed.runtime.types import Executor
from cubed.utils import convert_to_bytes


class Spec:
    """Specification of resources available to run a computation."""

    def __init__(
        self,
        work_dir: Union[str, None] = None,
        allowed_mem: Union[int, str, None] = None,
        reserved_mem: Union[int, str, None] = 0,
        executor: Union[Executor, None] = None,
        executor_name: Optional[str] = None,
        executor_options: Optional[dict] = None,
        storage_options: Union[dict, None] = None,
        zarr_compressor: Union[dict, str, None] = "default",
    ):
        """
        Specify resources available to run a computation.

        Parameters
        ----------
        work_dir : str or None
            The directory path (specified as an fsspec URL) used for storing intermediate data.
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
        zarr_compressor : dict or str, optional
            The compressor used by Zarr for intermediate data.

            If not specified, or set to ``"default"``, Zarr will use the default Blosc compressor.
            If set to ``None``, compression is disabled, which can be a good option when using local storage.
            Use a dictionary to configure arbitrary compression using Numcodecs. The following example specifies
            Blosc compression: ``zarr_compressor={"id": "blosc", "cname": "lz4", "clevel": 2, "shuffle": -1}``.
        """

        self._work_dir = work_dir

        self._reserved_mem = convert_to_bytes(reserved_mem or 0)
        if allowed_mem is None:
            self._allowed_mem = self.reserved_mem
        else:
            self._allowed_mem = convert_to_bytes(allowed_mem)

        self._executor = executor
        self._executor_name = executor_name
        self._executor_options = executor_options
        self._storage_options = storage_options
        self._zarr_compressor = zarr_compressor

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

    @cached_property
    def executor(self) -> Optional[Executor]:
        """The default executor for running computations."""
        if self._executor is not None:
            return self._executor
        elif self.executor_name is not None:
            return create_executor(self.executor_name, self.executor_options)
        return None

    @property
    def executor_name(self) -> Optional[str]:
        return self._executor_name

    @property
    def executor_options(self) -> Optional[dict]:
        return self._executor_options

    @property
    def storage_options(self) -> Optional[dict]:
        """Storage options to be passed to fsspec."""
        return self._storage_options

    @property
    def zarr_compressor(self) -> Union[dict, str, None]:
        """The compressor used by Zarr for intermediate data."""
        return self._zarr_compressor

    def __repr__(self) -> str:
        return (
            f"cubed.Spec(work_dir={self._work_dir}, allowed_mem={self._allowed_mem}, "
            f"reserved_mem={self._reserved_mem}, executor={self._executor}, storage_options={self._storage_options}, zarr_compressor={self._zarr_compressor})"
        )

    def __eq__(self, other):
        if isinstance(other, Spec):
            return (
                self.work_dir == other.work_dir
                and self.allowed_mem == other.allowed_mem
                and self.reserved_mem == other.reserved_mem
                and self.executor == other.executor
                and self.storage_options == other.storage_options
                and self.zarr_compressor == other.zarr_compressor
            )
        else:
            return False


def spec_from_config(config):
    return _spec_from_serialized_config(config.serialize())


@lru_cache  # ensure arrays have the same Spec object for a given config
def _spec_from_serialized_config(ser: str):
    config = donfig.deserialize(ser)
    spec_dict = expand_environment_variables(config["spec"])
    return Spec(**spec_dict)
