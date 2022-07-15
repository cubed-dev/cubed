import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal
from rechunker.executors.python import PythonPipelineExecutor

from cubed.primitive.blockwise import blockwise
from cubed.tests.utils import create_zarr, execute_pipeline


@pytest.fixture(scope="module", params=[PythonPipelineExecutor()])
def executor(request):
    return request.param


def test_blockwise(tmp_path, executor):
    source1 = create_zarr(
        [0, 1, 2], dtype=int, chunks=2, store=tmp_path / "source1.zarr"
    )
    source2 = create_zarr(
        [10, 50, 100], dtype=int, chunks=2, store=tmp_path / "source2.zarr"
    )
    max_mem = 1000
    target_store = tmp_path / "target.zarr"

    pipeline, target, required_mem, num_tasks = blockwise(
        np.outer,
        "ij",
        source1,
        "i",
        source2,
        "j",
        max_mem=max_mem,
        target_store=target_store,
        shape=(3, 3),
        dtype=int,
        chunks=(2, 2),
    )

    assert target.shape == (3, 3)
    assert target.dtype == int
    assert target.chunks == (2, 2)

    itemsize = np.dtype(int).itemsize
    assert required_mem == (
        (itemsize * 2)  # source1 compressed chunk
        + (itemsize * 2)  # source1 uncompressed chunk
        + (itemsize * 2)  # source2 compressed chunk
        + (itemsize * 2)  # source2 uncompressed chunk
        + (itemsize * 2 * 2)  # output compressed chunk
        + (itemsize * 2 * 2)  # output uncompressed chunk
    )

    assert num_tasks == 4

    execute_pipeline(pipeline, executor=executor)

    res = zarr.open(target_store)
    assert_array_equal(res[:], np.outer([0, 1, 2], [10, 50, 100]))


def _permute_dims(x, /, axes, max_mem, target_store):
    # From dask transpose
    if axes:
        if len(axes) != x.ndim:
            raise ValueError("axes don't match array")
    else:
        axes = tuple(range(x.ndim))[::-1]
    axes = tuple(d + x.ndim if d < 0 else d for d in axes)
    return blockwise(
        np.transpose,
        axes,
        x,
        tuple(range(x.ndim)),
        max_mem=max_mem,
        target_store=target_store,
        shape=x.shape,
        dtype=x.dtype,
        chunks=x.chunks,
        axes=axes,
    )


def test_blockwise_with_args(tmp_path, executor):
    source = create_zarr(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=int,
        chunks=(2, 2),
        store=tmp_path / "source.zarr",
    )
    max_mem = 1000
    target_store = tmp_path / "target.zarr"

    pipeline, target, required_mem, num_tasks = _permute_dims(
        source, axes=(1, 0), max_mem=max_mem, target_store=target_store
    )

    assert target.shape == (3, 3)
    assert target.dtype == int
    assert target.chunks == (2, 2)

    itemsize = np.dtype(int).itemsize
    assert required_mem == (
        (itemsize * 2 * 2)  # source compressed chunk
        + (itemsize * 2 * 2)  # source uncompressed chunk
        + (itemsize * 2 * 2)  # output compressed chunk
        + (itemsize * 2 * 2)  # output uncompressed chunk
    )

    assert num_tasks == 4

    execute_pipeline(pipeline, executor=executor)

    res = zarr.open(target_store)
    assert_array_equal(
        res[:], np.transpose(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), axes=(1, 0))
    )


def test_blockwise_max_mem_exceeded(tmp_path):
    source1 = create_zarr(
        [0, 1, 2], dtype=int, chunks=2, store=tmp_path / "source1.zarr"
    )
    source2 = create_zarr(
        [10, 50, 100], dtype=int, chunks=2, store=tmp_path / "source2.zarr"
    )
    max_mem = 100
    target_store = tmp_path / "target.zarr"

    with pytest.raises(
        ValueError, match=r"Blockwise memory \(\d+\) exceeds max_mem \(100\)"
    ):
        blockwise(
            np.outer,
            "ij",
            source1,
            "i",
            source2,
            "j",
            max_mem=max_mem,
            target_store=target_store,
            shape=(3, 3),
            dtype=int,
            chunks=(2, 2),
        )
