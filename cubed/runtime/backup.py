import math
from typing import Dict, TypeVar

T = TypeVar("T")


def should_launch_backup(
    task: T,
    now: float,
    start_times: Dict[T, float],
    end_times: Dict[T, float],
    min_tasks: int = 10,
    min_completed_fraction: float = 0.5,
    slow_factor: float = 3.0,
) -> bool:
    """
    Determine whether to launch a backup task.

    Backup tasks are only launched if there are at least `min_tasks` being run, and `min_completed_fraction` of tasks have completed.
    If both those criteria have been met, then a backup task is launched if the duration of the current task is at least
    `slow_factor` times slower than the `min_completed_fraction` percentile task duration.
    """
    if len(start_times) < min_tasks:
        return False
    n = math.ceil(len(start_times) * min_completed_fraction) - 1
    if len(end_times) <= n:
        return False
    completed_durations = sorted(
        [end_times[task] - start_times[task] for task in end_times]
    )
    duration = now - start_times[task]
    return duration > completed_durations[n] * slow_factor
