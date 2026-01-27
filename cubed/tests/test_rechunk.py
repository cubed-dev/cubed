import pytest

import cubed
import cubed as xp
from cubed._testing import assert_array_equal
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import _store_array, split_chunksizes
from cubed.core.rechunk import (
    calculate_regular_stage_chunks,
    multistage_regular_rechunking_plan,
    multspace,
    verify_chunk_compatibility,
)
from cubed.utils import itemsize
from cubed.vendor.rechunker.algorithm import (
    calculate_stage_chunks,
    multistage_rechunking_plan,
)


@pytest.mark.parametrize(
    (
        "min_mem",
        "expected_num_stages",
        "expected_max_input_blocks",
        "expected_max_output_blocks",
    ),
    [
        (None, 5, 7, 9),  # multistage rechunk - more stages, lower fan in/out
        (0, 1, 3771, 3460),  # single stage rechunk - very high fan in/out
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_rechunk_era5(
    tmp_path,
    min_mem,
    expected_num_stages,
    expected_max_input_blocks,
    expected_max_output_blocks,
):
    # from https://github.com/pangeo-data/rechunker/pull/89
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    spec = cubed.Spec(allowed_mem="2.5GB")

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)
    b = a.rechunk(target_chunks, min_mem=min_mem)

    b.visualize(filename=tmp_path / "rechunk_era5")

    assert b.shape == a.shape

    # find rechunk ops in plan
    plan = b.plan()
    rechunks = [
        (n, d)
        for (n, d) in plan.dag.nodes(data=True)
        if d.get("op_name", None) == "rechunk"
    ]

    # each stage has two ops due to intermediate store
    assert len(rechunks) == expected_num_stages * 2

    max_input_blocks = max(
        d["pipeline"].config.num_input_blocks[0] for _, d in rechunks
    )
    assert max_input_blocks == expected_max_input_blocks

    max_output_blocks = max(
        d["pipeline"].config.num_output_blocks[0] for _, d in rechunks
    )
    assert max_output_blocks == expected_max_output_blocks


@pytest.mark.parametrize(
    "spec",
    [
        cubed.Spec(allowed_mem="2.5GB"),
        # cloud stores use extra buffer copies, so need more memory for same rechunk plan
        cubed.Spec("s3://cubed-unittest/rechunk-era5", allowed_mem="3.5GB"),
    ],
)
def test_rechunk_era5_chunk_sizes(spec):
    # from https://github.com/pangeo-data/rechunker/pull/89
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    rechunk_plan = list(_rechunk_plan(a, target_chunks))
    assert rechunk_plan == [
        ((93, 721, 1440), (93, 240, 480)),
        ((465, 240, 480), (465, 240, 480)),
        ((465, 240, 480), (465, 120, 240)),
        ((2325, 120, 240), (2325, 120, 240)),
        ((2325, 120, 240), (2325, 40, 120)),
        ((11625, 40, 120), (11625, 40, 120)),
        ((11625, 40, 120), (11625, 20, 60)),
        ((58125, 20, 60), (58125, 20, 60)),
        ((58125, 20, 60), (58125, 10, 30)),
        ((350640, 10, 30), (350640, 10, 10)),
    ]


def test_rechunk_and_store():
    # from https://github.com/cubed-dev/cubed/issues/859
    shape = (394488, 778, 706)
    source_chunks = (24, 778, 706)
    target_chunks = (43800, 5, 5)
    spec = cubed.Spec(allowed_mem="5GB")

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)
    b = a.rechunk(target_chunks)

    num_tasks = b.plan().num_tasks

    # simulate what to_zarr or store does
    c = _store_array(b, None)

    c.visualize()

    # store should not change number of tasks
    assert c.plan().num_tasks == num_tasks


def test_rechunk_hypothesis_generated_bug():
    rechunk_shapes = (tuple([1001, 1001]), (38, 376), (5, 146))
    shape, source_chunks, target_chunks = rechunk_shapes

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    rechunk_plan = list(_rechunk_plan(a, target_chunks))
    assert rechunk_plan == [((30, 376), (15, 376)), ((15, 1001), (5, 146))]
    for copy_chunks, target_chunks in rechunk_plan:
        verify_chunk_compatibility(shape, copy_chunks, target_chunks)

    b = a.rechunk((5, 146))

    assert_array_equal(b.compute(), nxp.ones(shape))


def test_rechunk_hypothesis_generated_bug_allow_irregular():
    rechunk_shapes = (tuple([1001, 1001]), (38, 376), (5, 146))
    shape, source_chunks, target_chunks = rechunk_shapes

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    rechunk_plan = list(_rechunk_plan(a, target_chunks, allow_irregular=True))
    assert rechunk_plan == [((38, 376), (15, 376)), ((15, 1001), (5, 146))]

    b = a.rechunk((5, 146), allow_irregular=True)

    assert_array_equal(b.compute(), nxp.ones(shape))


@pytest.mark.parametrize(
    ("start", "stop", "num", "expected"),
    [
        (1, 1000, 2, [10, 100]),
        (1000, 1, 2, [100, 10]),
        (1, 1000, 0, []),
        (25, 25, 1, [25]),
        (24, 43800, 3, [144, 1008, 6048]),
    ],
)
def test_multspace(start, stop, num, expected):
    result = multspace(start, stop, num)

    assert len(result) == num
    assert result == expected

    # check that each value is a multiple of the smallest
    smallest = min(start, stop)
    assert all(val % smallest == 0 for val in result)


def test_calculate_stage_chunks():
    # era5
    # shape = (350640, 721, 1440)
    # source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    read_chunks = (93, 721, 1440)

    # cubed algorithm produces regular chunkings:
    # 1395 and 22320 are multiples of 93 (from read chunks)
    # 160 and 40 are multiples of 10 (from target chunks)
    # 250 and 50 are multiples of 10 (from target chunks)
    stage_chunks = calculate_regular_stage_chunks(
        read_chunks, target_chunks, stage_count=3
    )
    assert stage_chunks == [(1395, 160, 250), (22320, 40, 50)]

    # whereas the rechunker algorithm does not produce regular chunkings
    stage_chunks = calculate_stage_chunks(read_chunks, target_chunks, stage_count=3)
    assert stage_chunks == [(1447, 173, 274), (22528, 41, 52)]


def test_multistage_rechunking_plan():
    # era5
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)
    min_mem = 25000000
    max_mem = 500000000

    stages = multistage_rechunking_plan(
        shape, source_chunks, target_chunks, itemsize(xp.float32), min_mem, max_mem
    )
    assert stages == [
        ((93, 721, 1440), (93, 173, 396), (1447, 173, 396)),
        ((1447, 173, 396), (1447, 41, 109), (22528, 41, 109)),
        ((22528, 41, 109), (22528, 10, 30), (350640, 10, 30)),
    ]

    with pytest.raises(AssertionError, match="do not evenly slice target chunks"):  # noqa: PT012
        for stage in stages:
            read_chunks, int_chunks, write_chunks = stage
            verify_chunk_compatibility(shape, read_chunks, int_chunks)

    stages = multistage_regular_rechunking_plan(
        shape, source_chunks, target_chunks, itemsize(xp.float32), min_mem, max_mem
    )
    assert stages == [
        ((93, 721, 1440), (93, 240, 480), (465, 240, 480)),
        ((465, 240, 480), (465, 120, 240), (2325, 120, 240)),
        ((2325, 120, 240), (2325, 40, 120), (11625, 40, 120)),
        ((11625, 40, 120), (11625, 20, 60), (58125, 20, 60)),
        ((58125, 20, 60), (58125, 10, 30), (350640, 10, 30)),
    ]
    for i, stage in enumerate(stages):
        last_stage = i == len(stages) - 1
        read_chunks, int_chunks, write_chunks = stage
        verify_chunk_compatibility(shape, read_chunks, int_chunks)
        if last_stage:
            verify_chunk_compatibility(shape, write_chunks, target_chunks)


def test_split_chunksizes():
    assert split_chunksizes(10, 2, 4) == (2, 2, 2, 2, 2)
    assert split_chunksizes(10, 2, 3) == (2, 1, 1, 2, 2, 1, 1)
    # from example in _count_intermediate_chunks doc
    assert split_chunksizes(20, 5, 7) == (5, 2, 3, 4, 1, 5)
    # from hypothesis generated bug (above)
    assert split_chunksizes(1001, 38, 15)[:8] == (15, 15, 8, 7, 15, 15, 1, 14)
