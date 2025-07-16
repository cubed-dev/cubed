from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import memray
from memray._memray import compute_statistics
from memray._stats import Stats

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback


class AllocationType(Enum):
    # integer values match memray AllocatorType
    MALLOC = 1
    FREE = 2
    CALLOC = 3
    REALLOC = 4
    MMAP = 10
    MUNMAP = 11


@dataclass()
class Allocation:
    object_id: str
    allocation_type: AllocationType
    memory: int
    address: Optional[int] = None
    call: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.object_id} {self.allocation_type.name} {self.memory or ''} {self.address or ''} {self.call or ''}"


class MemrayCallback(Callback):
    """Process Memray results for a computation, and print large MALLOC and FREE calls for each operation."""

    def __init__(self, mem_threshold=50_000_000) -> None:
        self.mem_threshold = mem_threshold
        self.allocations: Dict[str, Allocation] = {}
        self.stats: Dict[str, Stats] = {}

    def on_compute_end(self, event):
        for name, _ in visit_nodes(event.dag):
            memray_result_file = f"history/{event.compute_id}/memray/{name}.bin"
            if not Path(memray_result_file).is_file():
                continue

            allocations = get_allocations_over_threshold(
                memray_result_file, self.mem_threshold
            )

            print(memray_result_file)
            for allocation in allocations:
                print(allocation)

            stats = compute_statistics(memray_result_file)
            print(f"Peak memory allocated: {stats.peak_memory_allocated}")

            print()

            self.allocations[name] = allocations
            self.stats[name] = stats


def get_allocations_over_threshold(result_file, mem_threshold):
    # find all allocations over threshold and their corresponding free operations
    id = 0
    address_to_allocation = {}
    with memray.FileReader(result_file) as reader:
        for a in reader.get_allocation_records():
            if a.size >= mem_threshold:
                func, mod, line = a.stack_trace()[0]
                if a.allocator == memray.AllocatorType.MALLOC:
                    allocation_type = AllocationType.MALLOC
                elif a.allocator == memray.AllocatorType.CALLOC:
                    allocation_type = AllocationType.CALLOC
                elif a.allocator == memray.AllocatorType.REALLOC:
                    allocation_type = AllocationType.REALLOC
                elif a.allocator == memray.AllocatorType.MMAP:
                    allocation_type = AllocationType.MMAP
                else:
                    raise ValueError(f"Unsupported memray.AllocatorType {a.allocator}")
                allocation = Allocation(
                    f"object-{id:03}",
                    allocation_type,
                    a.size,
                    address=a.address,
                    call=f"{func};{mod};{line}",
                )
                id += 1
                address_to_allocation[a.address] = allocation
                yield allocation
            elif (
                a.allocator in (memray.AllocatorType.FREE, memray.AllocatorType.MUNMAP)
                and a.address in address_to_allocation
            ):
                if a.allocator == memray.AllocatorType.FREE:
                    allocation_type = AllocationType.FREE
                elif a.allocator == memray.AllocatorType.MUNMAP:
                    allocation_type = AllocationType.MUNMAP
                else:
                    raise ValueError(f"Unsupported memray.AllocatorType {a.allocator}")
                allocation = address_to_allocation.pop(a.address)
                yield Allocation(
                    allocation.object_id,
                    allocation_type,
                    allocation.memory,
                    address=a.address,
                )
