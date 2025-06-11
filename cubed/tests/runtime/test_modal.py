import itertools

import pytest

modal = pytest.importorskip("modal")

import asyncio

import fsspec
import modal

from cubed.runtime.asyncio import async_map_unordered
from cubed.runtime.executors.modal import modal_create_futures_func
from cubed.tests.runtime.utils import check_invocation_counts, deterministic_failure

tmp_path = "s3://cubed-unittest/map_unordered"
region = "us-east-1"  # S3 region for above bucket

app = modal.App("cubed-test-app", include_source=True)

image = modal.Image.debian_slim().pip_install(
    [
        "array-api-compat",
        "donfig",
        "fsspec",
        "mypy_extensions",  # for rechunker
        "ndindex",
        "networkx",
        "pytest-mock",  # TODO: only needed for tests
        "s3fs",
        "tenacity",
        "toolz",
        "zarr",
    ]
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-aws-secret")],
    retries=2,
    timeout=10,
    cloud="aws",
    region=region,
)
def deterministic_failure_modal(i, path=None, timing_map=None, *, name=None):
    return deterministic_failure(path, timing_map, i, name=name)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-aws-secret")],
    timeout=10,
    cloud="aws",
    region=region,
)
def deterministic_failure_modal_no_retries(i, path=None, timing_map=None, *, name=None):
    return deterministic_failure(path, timing_map, i, name=name)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("my-aws-secret")],
    retries=2,
    timeout=300,
    cloud="aws",
    region=region,
)
def deterministic_failure_modal_long_timeout(
    i, path=None, timing_map=None, *, name=None
):
    return deterministic_failure(path, timing_map, i, name=name)


async def run_test(app_function, input, use_backups=False, batch_size=None, **kwargs):
    outputs = set()
    with modal.enable_output():
        async with app.run():
            create_futures_func = modal_create_futures_func(app_function)
            async for output in async_map_unordered(
                create_futures_func,
                input,
                use_backups=use_backups,
                batch_size=batch_size,
                **kwargs,
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
@pytest.mark.parametrize("use_backups", [False, True])
@pytest.mark.cloud
def test_success(timing_map, n_tasks, retries, use_backups):
    try:
        outputs = asyncio.run(
            run_test(
                app_function=deterministic_failure_modal,
                input=range(n_tasks),
                use_backups=use_backups,
                path=tmp_path,
                timing_map=timing_map,
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
@pytest.mark.parametrize("use_backups", [False, True])
@pytest.mark.cloud
def test_failure(timing_map, n_tasks, retries, use_backups):
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(
                run_test(
                    app_function=deterministic_failure_modal,
                    input=range(n_tasks),
                    use_backups=use_backups,
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
@pytest.mark.parametrize("use_backups", [False, True])
@pytest.mark.cloud
def test_large_number_of_tasks(timing_map, n_tasks, retries, use_backups):
    try:
        outputs = asyncio.run(
            run_test(
                app_function=deterministic_failure_modal,
                input=range(n_tasks),
                use_backups=use_backups,
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


@pytest.mark.cloud
def test_batch(tmp_path):
    # input is unbounded, so if entire input were consumed and not read
    # in batches then it would never return, since it would never
    # run the first (failing) input
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(
                run_test(
                    app_function=deterministic_failure_modal_no_retries,
                    input=itertools.count(),
                    path=tmp_path,
                    timing_map={0: [-1]},
                    batch_size=10,
                )
            )

    finally:
        fs = fsspec.open(tmp_path).fs
        fs.rm(tmp_path, recursive=True)
