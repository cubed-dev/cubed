import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal
from rechunker.executors.python import PythonPipelineExecutor

from cubed.primitive.rechunk import rechunk
from cubed.tests.utils import execute_pipeline


@pytest.fixture(scope="module", params=[PythonPipelineExecutor()])
def executor(request):
    return request.param


@pytest.mark.parametrize(
    "shape, source_chunks, max_mem, target_chunks, expected_required_mem, expected_num_tasks",
    [
        # one target chunk is made up of two source chunks
        (
            (4, 4),
            (1, 2),
            1000,
            (1, 4),
            1000,
            1,
        ),
        # only enough memory for one source/target chunk
        (
            (4, 4),
            (1, 4),
            4 * 8 * 4,  # elts x itemsize x copies
            (4, 1),
            4 * 8 * 4,  # elts x itemsize x copies
            8,
        ),
    ],
)
def test_rechunk(
    tmp_path,
    executor,
    shape,
    source_chunks,
    max_mem,
    target_chunks,
    expected_required_mem,
    expected_num_tasks,
):
    source = zarr.ones(shape, chunks=source_chunks, store=tmp_path / "source.zarr")
    max_mem = max_mem
    target_store = tmp_path / "target.zarr"
    temp_store = tmp_path / "temp.zarr"

    pipeline = rechunk(
        source,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )

    assert pipeline.target_array.shape == shape
    assert pipeline.target_array.dtype == source.dtype
    assert pipeline.target_array.chunks == target_chunks

    assert pipeline.required_mem == expected_required_mem

    assert pipeline.num_tasks == expected_num_tasks

    execute_pipeline(pipeline, executor=executor)

    res = zarr.open(target_store)
    assert_array_equal(res[:], np.ones(shape))
    assert res.chunks == target_chunks


def test_rechunk_max_mem_exceeded(tmp_path):
    source = zarr.ones((4, 4), chunks=(2, 2), store=tmp_path / "source.zarr")
    max_mem = 16
    target_store = tmp_path / "target.zarr"
    temp_store = tmp_path / "temp.zarr"

    # max mem is reduced by a factor of 4 from 16 to 4
    with pytest.raises(
        ValueError, match=r"Source chunk memory \(32\) exceeds max_mem \(4\)"
    ):
        rechunk(
            source,
            target_chunks=(4, 1),
            max_mem=max_mem,
            target_store=target_store,
            temp_store=temp_store,
        )
