import math


def should_launch_backup(
    task,
    now,
    start_times,
    end_times,
    min_tasks=10,
    min_completed_fraction=0.5,
    slow_factor=3.0,
):
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
