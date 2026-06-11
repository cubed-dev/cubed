from math import ceil, prod

import pytest

import cubed
import cubed as xp
from cubed._testing import assert_array_equal
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import _store_array, split_chunksizes
from cubed.core.rechunk import (
    RechunkPlanStats,
    _limit_chunks,
    calculate_regular_stage_chunks,
    multistage_regular_rechunking_plan,
    multspace,
    rechunk_plan,
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
        (None, 3, 16, 15),  # multistage rechunk - more stages, lower fan in/out
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

    # number of rechunk copy ops is one more than the number of stages
    assert len(rechunks) == expected_num_stages + 1

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
        ((93, 721, 1440), (93, 173, 396)),
        ((1447, 173, 396), (1447, 41, 109)),
        ((22528, 41, 109), (22528, 10, 30)),
        ((350640, 10, 30), (350640, 10, 10)),
    ]


@pytest.mark.parametrize("target", [None, "output.zarr"])
def test_rechunk_and_store(tmp_path, target):
    # from https://github.com/cubed-dev/cubed/issues/859
    shape = (394488, 778, 706)
    source_chunks = (24, 778, 706)
    target_chunks = (43800, 5, 5)
    spec = cubed.Spec(allowed_mem="5GB")

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)
    b = a.rechunk(target_chunks)

    num_tasks = b.plan().num_tasks

    # simulate what to_zarr or store does
    if target is not None:
        target = tmp_path / target
    c = _store_array(b, target)

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
    assert rechunk_plan == [((38, 376), (15, 376)), ((15, 1001), (5, 146))]

    b = a.rechunk((5, 146))

    assert_array_equal(b.compute(), nxp.ones(shape))


def test_rechunk_allow_irregular_false():
    # Verify the regular planner still works when allow_irregular=False is
    # passed explicitly. Regular intermediates must be integer multiples of
    # target chunks, so copy_chunks[0]=30 (multiple of 15) instead of 38.
    shape = (1001, 1001)
    source_chunks = (38, 376)
    target_chunks = (5, 146)

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    rechunk_plan = list(_rechunk_plan(a, target_chunks, allow_irregular=False))
    assert rechunk_plan == [((30, 376), (15, 376)), ((15, 1001), (5, 146))]
    for copy_chunks, store_chunks in rechunk_plan:
        verify_chunk_compatibility(shape, copy_chunks, store_chunks)

    b = a.rechunk(target_chunks, allow_irregular=False)

    assert_array_equal(b.compute(), nxp.ones(shape))


def test_rechunk_max_input_blocks():
    # Small workload where max_input_blocks successfully reduces fan-in.
    # Without limit: one stage, copy_chunks=(10,8,6), fan_in=8, fan_out=12, iops=20.
    # With max_input_blocks=5: write_chunks in dim 2 is shrunk so fan_in drops to 4.
    shape = (10, 8, 6)
    source_chunks = (5, 4, 3)
    target_chunks = (10, 2, 2)

    spec = cubed.Spec(allowed_mem="2.5GB")
    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    plan_no_limit = list(_rechunk_plan(a, target_chunks))
    assert plan_no_limit == [((10, 8, 6), (10, 2, 2))]

    b_no_limit = a.rechunk(target_chunks)
    stats_no_limit = RechunkPlanStats.from_plan(
        rechunk_plan(a, target_chunks), b_no_limit.plan()
    )
    assert stats_no_limit.num_copy_ops == 1
    assert stats_no_limit.max_task_input_blocks == 8
    assert stats_no_limit.max_task_output_blocks == 12
    assert stats_no_limit.max_task_iops == 20

    plan_limited = list(_rechunk_plan(a, target_chunks, max_input_blocks=5))
    assert plan_limited == [((10, 8, 2), (10, 2, 2))]

    b_limited = a.rechunk(target_chunks, max_input_blocks=5)
    stats_limited = RechunkPlanStats.from_plan(
        rechunk_plan(a, target_chunks, max_input_blocks=5), b_limited.plan()
    )
    assert stats_limited.num_copy_ops == 1
    # fan_in reduced to <= max_input_blocks=5
    assert stats_limited.max_task_input_blocks == 4
    assert stats_limited.max_task_output_blocks == 4
    assert stats_limited.max_task_iops == 8


def test_rechunk_max_output_blocks():
    shape = (2480, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (2480, 10, 10)

    spec = cubed.Spec(allowed_mem="2.5GB")
    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    # Without max_output_blocks: final op has large fan-out (lon axis fully consolidated)
    b = a.rechunk(target_chunks)
    stats_no_limit = RechunkPlanStats.from_plan(
        rechunk_plan(a, target_chunks), b.plan()
    )
    assert stats_no_limit.num_copy_ops == 3
    assert stats_no_limit.max_task_output_blocks > 400

    # With max_output_blocks=50: _limit_chunks brings fan-out within budget at stage 1.
    b_limited = a.rechunk(target_chunks, max_output_blocks=50)
    stats_limited = RechunkPlanStats.from_plan(
        rechunk_plan(a, target_chunks, max_output_blocks=50), b_limited.plan()
    )
    assert stats_limited.num_copy_ops == 2
    assert stats_limited.max_task_output_blocks <= 50


def test_rechunk_plan_viz():
    rechunk_shapes = (tuple([1001, 1001]), (38, 376), (5, 146))
    shape, source_chunks, target_chunks = rechunk_shapes

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)

    rplan = rechunk_plan(a, target_chunks)
    assert len(rplan.copy_ops) == 2
    # check generating the repr doesn't raise an exception
    rplan._repr_html_()

    # check stats (without running the rechunk)
    b = a.rechunk(target_chunks)
    plan = b.plan()
    stats = RechunkPlanStats.from_plan(rplan, plan)
    assert stats.num_copy_ops == 2
    assert stats.max_task_iops == 23


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


@pytest.mark.parametrize(
    ("chunks", "ratio_chunks", "align_to", "max_blocks", "expected"),
    [
        # max_blocks already satisfied
        pytest.param((4,), (4,), (4,), 1, (4,), id="within_budget_1d"),
        pytest.param((4, 4), (2, 2), (2, 2), 4, (4, 4), id="within_budget_2d"),
        # fan-out reduction (ratio_chunks == align_to = target_chunks):
        #   ratios=[4] > 2; new_ratio=2 -> new_cc=6; next total=2 <= 2
        pytest.param((12,), (3,), (3,), 2, (6,), id="fan_out_1d"),
        #   ratios=[4,3]=12 > 6; dim 0 shrinks: new_ratio=2 -> new_cc=4; next total=6 <= 6
        pytest.param((8, 6), (2, 2), (2, 2), 6, (4, 6), id="fan_out_2d"),
        # fan-in reduction (ratio_chunks != align_to: source_chunks vs store_chunks):
        #   ratios=[ceil(10/3)=4] > 2; new_ratio=2 -> new_cc=(2*3//2)*2=6; next total=2 <= 2
        pytest.param((10,), (3,), (2,), 2, (6,), id="fan_in_1d"),
        #   ratios=[ceil(12/5)=3, ceil(6/3)=2]=6 > 4; dim 0: new_ratio=2 -> new_cc=(2*5//2)*2=10; next total=4 <= 4
        pytest.param((12, 6), (5, 3), (2, 2), 4, (10, 6), id="fan_in_2d"),
        # cannot achieve the max_blocks limit, return input unchanged
        pytest.param((3,), (2,), (3,), 1, (3,), id="structural_1d"),
        pytest.param((4, 4), (3, 3), (4, 4), 1, (4, 4), id="structural_2d"),
        pytest.param((6, 4), (4, 4), (6, 4), 1, (6, 4), id="structural_2d_mixed_ratio"),
    ],
)
def test_limit_chunks(chunks, ratio_chunks, align_to, max_blocks, expected):
    result = _limit_chunks(chunks, ratio_chunks, align_to, max_blocks)
    assert result == expected
    # output must remain a multiple of each align_to element
    for r, a in zip(result, align_to):
        assert r % a == 0
    # when reduction occurred the limit must now be satisfied
    if result != chunks:
        assert prod(ceil(r / rc) for r, rc in zip(result, ratio_chunks)) <= max_blocks
