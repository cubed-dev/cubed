from dataclasses import dataclass
from typing import List, Optional

from cubed.spec import Spec
from cubed.utils import is_cloud_storage_path


@dataclass
class BufferCopies:
    """Information about the number of buffer copies incurred for array storage operations."""

    read: int
    """The number of copies made when reading an array from storage."""

    write: int
    """The number of copies made when writing an array to storage."""


def get_buffer_copies(spec: Optional[Spec]):
    """Return the number of buffer copies to use, based on the spec.

    Using cloud storage will result in more buffer copies being accounted for.
    """

    # See https://github.com/tomwhite/memray-array
    # More factors (e.g. compression) could be taken into account in the future.

    if (
        spec is not None
        and spec.work_dir is not None
        and is_cloud_storage_path(spec.work_dir)
    ):
        return BufferCopies(read=2, write=2)

    return BufferCopies(read=1, write=1)


def calculate_projected_mem(
    reserved_mem: int,
    inputs: List[int],
    operation: int,
    output: int,
    buffer_copies: BufferCopies,
) -> int:
    """Calculate the projected memory needed to run an array operation.

    All memory sizes are in bytes.

    Parameters
    ----------
    reserved_mem : int
        The memory reserved on a worker for non-data use when running a task, in bytes.
    inputs : list of int
        The sizes of the input arrays, in bytes.
    operation : int
        Extra memory needed for the operation (not including allocating the output), in bytes.
    output : int
        The size of the output array, in bytes.
    buffer_copies: BufferCopies
        The number of buffer copies for reading an writing.

    Returns
    -------
    The projected memory needed to run an array operation.
    """

    projected_mem = reserved_mem

    for input in inputs:
        projected_mem += input * buffer_copies.read
        projected_mem += input

    projected_mem += operation

    projected_mem += output
    projected_mem += output * buffer_copies.write

    return projected_mem


class MemoryModeller:
    """Models peak memory usage for a series of operations."""

    current_mem: int = 0
    peak_mem: int = 0

    def allocate(self, num_bytes):
        self.current_mem += num_bytes
        self.peak_mem = max(self.peak_mem, self.current_mem)

    def free(self, num_bytes):
        self.current_mem -= num_bytes
        self.peak_mem = max(self.peak_mem, self.current_mem)
