import argparse
import logging

from apache_beam.options.pipeline_options import PipelineOptions

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.runtime.executors.beam import BeamExecutor


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmp_path",
        dest="tmp_path",
        help="Temporary path for intermediate Zarr arrays.",
    )
    known_args, pipeline_args = parser.parse_known_args(argv)
    beam_options = PipelineOptions(pipeline_args)

    spec = cubed.Spec(known_args.tmp_path, allowed_mem=2_000_000_000)
    executor = BeamExecutor()

    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)
    # use store=None to write to temporary zarr
    cubed.to_zarr(e, store=None, executor=executor, options=beam_options)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
