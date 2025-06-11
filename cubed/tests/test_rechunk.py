import pytest

import cubed
import cubed as xp


@pytest.mark.parametrize(
    "min_mem, expected_num_stages, expected_max_input_blocks, expected_max_output_blocks",
    [
        [None, 3, 16, 15],  # multistage rechunk - more stages, lower fan in/out
        [0, 1, 3771, 3460],  # single stage rechunk - very high fan in/out
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
    plan = b.plan._finalize()
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

    assert list(_rechunk_plan(a, target_chunks)) == [
        ((93, 721, 1440), (93, 173, 396)),
        ((1447, 173, 396), (1447, 173, 396)),
        ((1447, 173, 396), (1447, 41, 109)),
        ((22528, 41, 109), (22528, 41, 109)),
        ((22528, 41, 109), (22528, 10, 30)),
        ((350640, 10, 30), (350640, 10, 10)),
    ]
