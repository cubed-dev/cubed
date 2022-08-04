from cubed.runtime.backup import should_launch_backup


def test_should_launch_backup():
    start_times = {f"task-{task}": 0 for task in range(10)}
    end_times = {}

    # Don't launch a backup if nothing has completed yet
    assert not should_launch_backup("task-5", 7, start_times, end_times)

    end_times = {f"task-{task}": 4 for task in range(5)}

    # Don't launch a backup if not sufficiently slow (7s is not 3 times slower than 4s)
    assert not should_launch_backup("task-5", 7, start_times, end_times)

    # Don't launch a backup even if sufficiently slow (13s is 3 times slower than 4s) if not enough tasks
    assert not should_launch_backup("task-5", 13, start_times, end_times, min_tasks=20)

    # Launch a backup if sufficiently slow (13s is 3 times slower than 4s)
    assert should_launch_backup("task-5", 13, start_times, end_times)
