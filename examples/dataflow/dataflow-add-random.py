import argparse
import logging

from apache_beam.options.pipeline_options import PipelineOptions
from tqdm.contrib.logging import logging_redirect_tqdm

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.diagnostics.tqdm import TqdmProgressBar
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
    c = xp.add(a, b)
    with logging_redirect_tqdm():
        progress = TqdmProgressBar()
        # use store=None to write to temporary zarr
        cubed.to_zarr(
            c, store=None, executor=executor, callbacks=[progress], options=beam_options
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
