import pytest

modal = pytest.importorskip("modal")

import asyncio

import fsspec
import modal.aio

from cubed.runtime.executors.modal_async import map_unordered
from cubed.utils import join_path


def read_int_from_file(path):
    with fsspec.open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with fsspec.open(path, "w") as f:
        f.write(str(i))


tmp_path = "s3://cubed-unittest/map_unordered"


stub = modal.aio.AioStub()

image = modal.DebianSlim().pip_install(
    [
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


@stub.function(image=image, secret=modal.ref("my-aws-secret"))
def never_fail(i):
    invocation_count_file = join_path(tmp_path, f"{i}")
    write_int_to_file(invocation_count_file, 1)
    return i


@stub.function(image=image, secret=modal.ref("my-aws-secret"), retries=2)
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


@stub.function(image=image, secret=modal.ref("my-aws-secret"))
async def fail_on_first_invocation_no_retry(i):
    invocation_count_file = join_path(tmp_path, f"{i}")
    fs = fsspec.open(invocation_count_file).fs
    if fs.exists(invocation_count_file):
        count = read_int_from_file(invocation_count_file)
        write_int_to_file(invocation_count_file, count + 1)
    else:
        write_int_to_file(invocation_count_file, 1)
        raise RuntimeError(f"Fail on {i}")
    return i


@stub.function(image=image, secret=modal.ref("my-aws-secret"))
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


async def run_test(app_function, input, max_failures=3, use_backups=False):
    async with stub.run():
        async for _ in map_unordered(
            app_function, input, max_failures=max_failures, use_backups=use_backups
        ):
            pass


@pytest.mark.cloud
def test_map_unordered_no_failures():
    try:
        asyncio.run(run_test(app_function=never_fail, input=range(3)))

        assert read_int_from_file(join_path(tmp_path, "0")) == 1
        assert read_int_from_file(join_path(tmp_path, "1")) == 1
        assert read_int_from_file(join_path(tmp_path, "2")) == 1

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_unordered_recovers_from_failures():
    try:
        asyncio.run(run_test(app_function=fail_on_first_invocation, input=range(3)))

        assert read_int_from_file(join_path(tmp_path, "0")) == 2
        assert read_int_from_file(join_path(tmp_path, "1")) == 2
        assert read_int_from_file(join_path(tmp_path, "2")) == 2

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_unordered_too_many_failures():
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(
                run_test(
                    app_function=fail_on_first_invocation_no_retry,
                    input=range(3),
                )
            )

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


@pytest.mark.cloud
def test_map_unordered_stragglers():
    try:
        asyncio.run(
            run_test(
                app_function=sleep_on_first_invocation,
                input=range(10),
                use_backups=True,
            )
        )

        assert read_int_from_file(join_path(tmp_path, "0")) == 2
        for i in range(1, 10):
            assert read_int_from_file(join_path(tmp_path, f"{i}")) == 1

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)
