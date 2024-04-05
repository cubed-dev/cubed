import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from cubed.primitive.rechunk import rechunk
from cubed.runtime.executors.local import SingleThreadedExecutor
from cubed.storage.backend import open_backend_array
from cubed.tests.utils import execute_pipeline


@pytest.fixture(scope="module", params=[SingleThreadedExecutor()])
def executor(request):
    return request.param


@pytest.mark.parametrize(
    "shape, source_chunks, allowed_mem, reserved_mem, target_chunks, expected_projected_mem, expected_num_tasks",
    [
        # one target chunk is made up of two source chunks
        (
            (4, 4),
            (1, 2),
            1000,
            0,
            (1, 4),
            1000,
            (1,),
        ),
        # everything still works with 100 bytes of reserved_mem
        (
            (4, 4),
            (1, 2),
            1000,
            100,
            (1, 4),
            1000,
            (1,),
        ),
        # only enough memory for one source/target chunk
        (
            (4, 4),
            (1, 4),
            4 * 8 * 4,  # elts x itemsize x copies
            0,
            (4, 1),
            4 * 8 * 4,  # elts x itemsize x copies
            (16, 4),
        ),
    ],
)
def test_rechunk(
    tmp_path,
    executor,
    shape,
    source_chunks,
    allowed_mem,
    reserved_mem,
    target_chunks,
    expected_projected_mem,
    expected_num_tasks,
):
    source = zarr.ones(shape, chunks=source_chunks, store=tmp_path / "source.zarr")
    target_store = tmp_path / "target.zarr"
    temp_store = tmp_path / "temp.zarr"

    ops = rechunk(
        source,
        source_array_name="source-array",
        int_array_name="int-array",
        target_chunks=target_chunks,
        allowed_mem=allowed_mem,
        reserved_mem=reserved_mem,
        target_store=target_store,
        temp_store=temp_store,
    )

    assert len(ops) == len(expected_num_tasks)

    for i, op in enumerate(ops):
        assert op.target_array.shape == shape
        assert op.target_array.dtype == source.dtype

        assert op.projected_mem == expected_projected_mem

        assert op.num_tasks == expected_num_tasks[i]

    last_op = ops[-1]
    assert last_op.target_array.chunks == target_chunks

    # create lazy zarr arrays
    for op in ops:
        op.target_array.create()

    for op in ops:
        execute_pipeline(op.pipeline, executor=executor)

    res = open_backend_array(target_store, mode="r")
    assert_array_equal(res[:], np.ones(shape))
    assert res.chunks == target_chunks


def test_rechunk_allowed_mem_exceeded(tmp_path):
    source = zarr.ones((4, 4), chunks=(2, 2), store=tmp_path / "source.zarr")
    allowed_mem = 16
    target_store = tmp_path / "target.zarr"
    temp_store = tmp_path / "temp.zarr"

    # cubed's allowed_mem is reduced by a factor of 4 for rechunker's max_mem from 16 to 4
    with pytest.raises(
        ValueError, match=r"Source chunk memory \(32\) exceeds max_mem \(4\)"
    ):
        rechunk(
            source,
            source_array_name="source-array",
            int_array_name="int-array",
            target_chunks=(4, 1),
            allowed_mem=allowed_mem,
            reserved_mem=0,
            target_store=target_store,
            temp_store=temp_store,
        )
