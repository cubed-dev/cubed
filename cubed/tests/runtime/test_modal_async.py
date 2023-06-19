import pytest

modal = pytest.importorskip("modal")

import asyncio

import fsspec
import modal

from cubed.runtime.executors.modal_async import map_unordered
from cubed.tests.runtime.utils import check_invocation_counts, deterministic_failure

tmp_path = "s3://cubed-unittest/map_unordered"


stub = modal.Stub("cubed-test-stub")

image = modal.Image.debian_slim().pip_install(
    [
        "fsspec",
        "mypy_extensions",  # for rechunker
        "networkx",
        "pytest-mock",  # TODO: only needed for tests
        "s3fs",
        "tenacity",
        "toolz",
        "zarr",
    ]
)


@stub.function(
    image=image, secret=modal.Secret.from_name("my-aws-secret"), retries=2, timeout=10
)
def deterministic_failure_modal(i, path=None, timing_map=None):
    return deterministic_failure(path, timing_map, i)


@stub.function(
    image=image, secret=modal.Secret.from_name("my-aws-secret"), retries=2, timeout=300
)
def deterministic_failure_modal_long_timeout(i, path=None, timing_map=None):
    return deterministic_failure(path, timing_map, i)


async def run_test(app_function, input, use_backups=False, **kwargs):
    outputs = set()
    async with stub.run():
        async for output in map_unordered(
            app_function, input, use_backups=use_backups, **kwargs
        ):
            outputs.add(output)
    return outputs


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        # no failures
        ({}, 3, 2),
        # first invocation fails
        ({0: [-1], 1: [-1], 2: [-1]}, 3, 2),
        # first two invocations fail
        ({0: [-1, -1], 1: [-1, -1], 2: [-1, -1]}, 3, 2),
        # first input sleeps once
        ({0: [20]}, 3, 2),
    ],
)
# fmt: on
@pytest.mark.cloud
def test_success(timing_map, n_tasks, retries):
    try:
        outputs = asyncio.run(
            run_test(
                app_function=deterministic_failure_modal,
                input=range(n_tasks),
                path=tmp_path,
                timing_map=timing_map
            )
        )

        assert outputs == set(range(n_tasks))
        check_invocation_counts(tmp_path, timing_map, n_tasks, retries)

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        # too many failures
        ({0: [-1], 1: [-1], 2: [-1, -1, -1]}, 3, 2),
    ],
)
# fmt: on
@pytest.mark.cloud
def test_failure(timing_map, n_tasks, retries):
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(
                run_test(
                    app_function=deterministic_failure_modal,
                    input=range(n_tasks),
                    path=tmp_path,
                    timing_map=timing_map,
                )
            )

        check_invocation_counts(tmp_path, timing_map, n_tasks, retries)

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries",
    [
        ({0: [-1], 1: [-1], 2: [-1]}, 1000, 2),
    ],
)
# fmt: on
@pytest.mark.cloud
def test_large_number_of_tasks(timing_map, n_tasks, retries):
    try:
        outputs = asyncio.run(
            run_test(
                app_function=deterministic_failure_modal,
                input=range(n_tasks),
                path=tmp_path,
                timing_map=timing_map
            )
        )

        assert outputs == set(range(n_tasks))
        check_invocation_counts(tmp_path, timing_map, n_tasks, retries)

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)


# fmt: off
@pytest.mark.parametrize(
    "timing_map, n_tasks, retries, expected_invocation_counts_overrides",
    [
        # backup succeeds quickly
        ({0: [60]}, 10, 2, {0: 2}),
    ],
)
# fmt: on
@pytest.mark.cloud
def test_stragglers(timing_map, n_tasks, retries, expected_invocation_counts_overrides):
    try:
        outputs = asyncio.run(
            run_test(
                app_function=deterministic_failure_modal_long_timeout,
                input=range(n_tasks),
                path=tmp_path,
                timing_map=timing_map,
                use_backups=True,
            )
        )

        assert outputs == set(range(n_tasks))
        check_invocation_counts(tmp_path, timing_map, n_tasks, retries, expected_invocation_counts_overrides)

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)
