import asyncio

import fsspec
import modal
import modal.aio
import pytest

from cubed.runtime.executors.modal import launch_backup, map_as_completed
from cubed.utils import join_path


def read_int_from_file(path):
    with fsspec.open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with fsspec.open(path, "w") as f:
        f.write(str(i))


tmp_path = "s3://cubed-unittest/map_as_completed"


app = modal.aio.AioApp()

image = modal.DebianSlim(
    python_packages=[
        "dask[array]",
        "fsspec",
        "networkx",
        "pytest-mock",  # TODO: only needed for tests
        "rechunker",
        "s3fs",
        "tenacity",
        "zarr",
    ]
)


@app.function(image=image, secret=modal.ref("my-aws-secret"))
def never_fail(i):
    invocation_count_file = join_path(tmp_path, f"{i}")
    write_int_to_file(invocation_count_file, 1)
    return i


@app.function(image=image, secret=modal.ref("my-aws-secret"))
async def fail_on_first_invocation(i):
    invocation_count_file = join_path(tmp_path, f"{i}")
    fs = fsspec.open(invocation_count_file).fs
    if fs.exists(invocation_count_file):
        count = read_int_from_file(invocation_count_file)
        write_int_to_file(invocation_count_file, count + 1)
    else:
        write_int_to_file(invocation_count_file, 1)
        raise RuntimeError(f"Fail on {i}")
    return i


@app.function(image=image, secret=modal.ref("my-aws-secret"))
async def sleep_on_first_invocation(i):
    print(f"sleep_on_first_invocation {i}")
    invocation_count_file = join_path(tmp_path, f"{i}")
    fs = fsspec.open(invocation_count_file).fs
    if fs.exists(invocation_count_file):
        count = read_int_from_file(invocation_count_file)
        write_int_to_file(invocation_count_file, count + 1)
    else:
        write_int_to_file(invocation_count_file, 1)
        # only sleep on first invocation of input = 0
        if i == 0:
            print("sleeping...")
            await asyncio.sleep(60)
            print("... finished sleeping")
    return i


async def run_test(app_function, input, max_failures=3):
    async with app.run():
        async for _ in map_as_completed(app_function, input, max_failures=max_failures):
            pass


@pytest.mark.cloud
def test_map_as_completed_no_failures():
    try:
        asyncio.run(run_test(app_function=never_fail, input=range(3), max_failures=0))

        assert read_int_from_file(join_path(tmp_path, "0")) == 1
        assert read_int_from_file(join_path(tmp_path, "1")) == 1
        assert read_int_from_file(join_path(tmp_path, "2")) == 1

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_as_completed_recovers_from_failures():
    try:
        asyncio.run(run_test(app_function=fail_on_first_invocation, input=range(3)))

        assert read_int_from_file(join_path(tmp_path, "0")) == 2
        assert read_int_from_file(join_path(tmp_path, "1")) == 2
        assert read_int_from_file(join_path(tmp_path, "2")) == 2

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_as_completed_too_many_failures():
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(
                run_test(
                    app_function=fail_on_first_invocation,
                    input=range(3),
                    max_failures=2,
                )
            )

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_as_completed_stragglers():
    try:
        asyncio.run(run_test(app_function=sleep_on_first_invocation, input=range(10)))

        assert read_int_from_file(join_path(tmp_path, "0")) == 2
        for i in range(1, 10):
            assert read_int_from_file(join_path(tmp_path, f"{i}")) == 1

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


def test_launch_backup():
    start_times = {f"task-{task}": 0 for task in range(10)}
    end_times = {}

    # Don't launch a backup if nothing has completed yet
    assert not launch_backup("task-5", 7, start_times, end_times)

    end_times = {f"task-{task}": 4 for task in range(5)}

    # Don't launch a backup if not sufficiently slow (7s is not 3 times slower than 4s)
    assert not launch_backup("task-5", 7, start_times, end_times)

    # Launch a backup if sufficiently slow (13s is 3 times slower than 4s)
    assert launch_backup("task-5", 13, start_times, end_times)
