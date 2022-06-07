import asyncio

import fsspec
import modal
import modal.aio
import pytest

from cubed.runtime.executors.modal import map_with_retries
from cubed.utils import join_path


def read_int_from_file(path):
    with fsspec.open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with fsspec.open(path, "w") as f:
        f.write(str(i))


tmp_path = "s3://cubed-unittest/map_with_retries"


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


async def run_test(app_function, max_failures=3):
    async with app.run():
        await map_with_retries(app_function, [0, 1, 2], max_failures=max_failures)


@pytest.mark.cloud
def test_map_with_retries_no_failures():
    try:
        asyncio.run(run_test(app_function=never_fail, max_failures=0))

        assert read_int_from_file(join_path(tmp_path, "0")) == 1
        assert read_int_from_file(join_path(tmp_path, "1")) == 1
        assert read_int_from_file(join_path(tmp_path, "2")) == 1

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_with_retries_recovers_from_failures():
    try:
        asyncio.run(run_test(app_function=fail_on_first_invocation))

        assert read_int_from_file(join_path(tmp_path, "0")) == 2
        assert read_int_from_file(join_path(tmp_path, "1")) == 2
        assert read_int_from_file(join_path(tmp_path, "2")) == 2

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_with_retries_too_many_failures():
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(run_test(app_function=fail_on_first_invocation, max_failures=2))

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)
