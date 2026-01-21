import pytest

import cubed
import cubed as xp
from cubed.core.rechunk import calculate_stage_chunks as cubed_calculate_stage_chunks
from cubed.core.rechunk import (
    multistage_rechunking_plan as cubed_multistage_rechunking_plan,
)
from cubed.core.rechunk import multspace
from cubed.utils import itemsize
from cubed.vendor.rechunker.algorithm import (
    calculate_stage_chunks as rechunker_calculate_stage_chunks,
)
from cubed.vendor.rechunker.algorithm import (
    multistage_rechunking_plan as rechunker_multistage_rechunking_plan,
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
    b = a.rechunk(target_chunks, min_mem=min_mem, use_new_impl=True)

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
        ((93, 721, 1440), (93, 173, 396)),
        ((1447, 173, 396), (1447, 173, 396)),
        ((1447, 173, 396), (1447, 41, 109)),
        ((22528, 41, 109), (22528, 41, 109)),
        ((22528, 41, 109), (22528, 10, 30)),
        ((350640, 10, 30), (350640, 10, 10)),
    ]

    # like _verify_single_stage_plan_correctness in rechunker
    for copy_chunks, target_chunks in rechunk_plan:
        print(copy_chunks, target_chunks)
        for n, cc, tc in zip(shape, copy_chunks, target_chunks):
            assert (cc == n) or (cc % tc == 0), (
                f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
                f"since {cc} is not a multiple of {tc}"
            )


def test_rechunk_issue_859():
    # from https://github.com/cubed-dev/cubed/issues/859
    shape = (394488, 778, 706)
    source_chunks = (24, 778, 706)
    target_chunks = (43800, 5, 5)

    spec = cubed.Spec(allowed_mem="5GB")

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    rechunk_plan = list(_rechunk_plan(a, target_chunks))
    assert rechunk_plan == [
        ((432, 778, 706), (432, 62, 705)),
        ((4349, 62, 705), (4349, 62, 705)),
        ((4349, 62, 705), (4349, 5, 705)),
        ((43800, 5, 706), (43800, 5, 5)),
    ]

    # like _verify_single_stage_plan_correctness in rechunker
    for copy_chunks, target_chunks in rechunk_plan:
        print(copy_chunks, target_chunks)
        for n, cc, tc in zip(shape, copy_chunks, target_chunks):
            assert (cc == n) or (cc % tc == 0), (
                f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
                f"since {cc} is not a multiple of {tc}"
            )


def test_rechunk_issue_859_workaround():
    # from https://github.com/cubed-dev/cubed/issues/859
    shape = (394488, 778, 706)
    source_chunks = (24, 778, 706)
    target_chunks = (43800, 5, 5)

    intermediate_chunks = (3650, 50, 50)

    spec = cubed.Spec(allowed_mem="5GB")

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan

    rechunk_plan1 = list(_rechunk_plan(a, intermediate_chunks))

    # like _verify_single_stage_plan_correctness in rechunker
    for copy_chunks, target_chunks in rechunk_plan1:
        print(copy_chunks, target_chunks)
        for n, cc, tc in zip(shape, copy_chunks, target_chunks):
            assert (cc == n) or (cc % tc == 0), (
                f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
                f"since {cc} is not a multiple of {tc}"
            )

    b = xp.empty(shape, dtype=xp.float32, chunks=intermediate_chunks, spec=spec)

    rechunk_plan2 = list(_rechunk_plan(b, target_chunks))
    # like _verify_single_stage_plan_correctness in rechunker
    for copy_chunks, target_chunks in rechunk_plan2:
        print(copy_chunks, target_chunks)
        for n, cc, tc in zip(shape, copy_chunks, target_chunks):
            assert (cc == n) or (cc % tc == 0), (
                f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
                f"since {cc} is not a multiple of {tc}"
            )


@pytest.mark.parametrize(
    "spec",
    [
        cubed.Spec(allowed_mem="2.5GB"),
        # cloud stores use extra buffer copies, so need more memory for same rechunk plan
        # cubed.Spec("s3://cubed-unittest/rechunk-era5", allowed_mem="3.5GB"),
    ],
)
def test_rechunk_era5_chunk_sizes_new_cubed_algo(spec):
    # from https://github.com/pangeo-data/rechunker/pull/89
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    a = xp.empty(shape, dtype=xp.float32, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan2

    # TODO: can adjust min_mem=20000000 to get same number of stages as before
    rechunk_plan = list(_rechunk_plan2(a, target_chunks))
    assert rechunk_plan == [
        ((93, 721, 1440), (93, 270, 540)),
        ((744, 270, 540), (744, 270, 540)),
        ((744, 270, 540), (744, 90, 180)),
        ((5952, 90, 180), (5952, 90, 180)),
        ((5952, 90, 180), (5952, 30, 90)),
        ((47616, 30, 90), (47616, 30, 90)),
        ((47616, 30, 90), (47616, 10, 30)),
        ((350640, 10, 30), (350640, 10, 10)),
    ]

    # like _verify_single_stage_plan_correctness in rechunker
    for copy_chunks, target_chunks in rechunk_plan:
        print(copy_chunks, target_chunks)
        for n, cc, tc in zip(shape, copy_chunks, target_chunks):
            assert (cc == n) or (cc % tc == 0), (
                f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
                f"since {cc} is not a multiple of {tc}"
            )


def test_rechunk_hypothesis_generated_bug():
    rechunk_shapes = (tuple([1001, 1001]), (38, 376), (5, 146))
    shape, source_chunks, target_chunks = rechunk_shapes

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)

    from cubed.core.ops import _rechunk_plan, _rechunk_plan2, _rechunk_plan3

    rechunk_plan = list(_rechunk_plan(a, target_chunks))
    assert rechunk_plan == [((38, 376), (15, 376)), ((15, 1001), (5, 146))]

    # like _verify_single_stage_plan_correctness in rechunker

    # TODO: assert that this fails

    # for copy_chunks, target_chunks in rechunk_plan:
    #     print(copy_chunks, target_chunks)
    #     for n, cc, tc in zip(shape, copy_chunks, target_chunks):
    #         assert (cc == n) or (cc % tc == 0), (
    #             f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
    #             f"since {cc} is not a multiple of {tc}"
    #         )

    rechunk_plan = list(_rechunk_plan2(a, target_chunks))
    assert rechunk_plan == [((38, 376), (15, 376)), ((15, 1001), (5, 146))]

    rechunk_plan = list(_rechunk_plan3(a, target_chunks))
    assert rechunk_plan == [((30, 376), (15, 376)), ((15, 1001), (5, 146))]

    # like _verify_single_stage_plan_correctness in rechunker
    for copy_chunks, target_chunks in rechunk_plan:
        print(copy_chunks, target_chunks)
        for n, cc, tc in zip(shape, copy_chunks, target_chunks):
            assert (cc == n) or (cc % tc == 0), (
                f"copy chunks {copy_chunks} do not evenly slice target chunks {target_chunks}, "
                f"since {cc} is not a multiple of {tc}"
            )


def test_rechunker_calculate_stage_chunks():
    # era5
    # shape = (350640, 721, 1440)
    # source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    read_chunks = (93, 721, 1440)
    stage_chunks = rechunker_calculate_stage_chunks(
        read_chunks, target_chunks, stage_count=3
    )
    assert stage_chunks == [(1447, 173, 274), (22528, 41, 52)]


def test_cubed_calculate_stage_chunks():
    # era5
    # shape = (350640, 721, 1440)
    # source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)

    read_chunks = (93, 721, 1440)
    stage_chunks = cubed_calculate_stage_chunks(
        read_chunks, target_chunks, stage_count=3
    )
    assert stage_chunks == [(1488, 160, 250), (22320, 40, 50)]


def test_rechunker_multistage_rechunking_plan():
    # era5
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)
    min_mem = 25000000
    max_mem = 500000000
    stages = rechunker_multistage_rechunking_plan(
        shape, source_chunks, target_chunks, itemsize(xp.float32), min_mem, max_mem
    )

    assert stages == [
        ((93, 721, 1440), (93, 173, 396), (1447, 173, 396)),
        ((1447, 173, 396), (1447, 41, 109), (22528, 41, 109)),
        ((22528, 41, 109), (22528, 10, 30), (350640, 10, 30)),
    ]


def test_cubed_multistage_rechunking_plan():
    # era5
    shape = (350640, 721, 1440)
    source_chunks = (31, 721, 1440)
    target_chunks = (350640, 10, 10)
    min_mem = 25000000
    max_mem = 500000000
    stages = cubed_multistage_rechunking_plan(
        shape, source_chunks, target_chunks, itemsize(xp.float32), min_mem, max_mem
    )

    assert stages == [
        ((93, 721, 1440), (93, 173, 396), (1447, 173, 396)),
        ((1447, 173, 396), (1447, 41, 109), (22528, 41, 109)),
        ((22528, 41, 109), (22528, 10, 30), (350640, 10, 30)),
    ]


def test_multspace():
    a = multspace(24, 43800, 5)
    prev = None
    for ai in a:
        if prev is not None:
            assert ai % prev == 0
        prev = ai

    a = multspace(43800, 24, 5)
    prev = None
    for ai in a[::-1]:
        if prev is not None:
            assert ai % prev == 0
        prev = ai
