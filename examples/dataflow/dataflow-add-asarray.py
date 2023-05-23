import argparse
import logging

from apache_beam.options.pipeline_options import PipelineOptions

import cubed
import cubed.array_api as xp
from cubed.runtime.executors.beam import BeamDagExecutor


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmp_path",
        dest="tmp_path",
        help="Temporary path for intermediate Zarr arrays.",
    )
    known_args, pipeline_args = parser.parse_known_args(argv)
    beam_options = PipelineOptions(pipeline_args)

    spec = cubed.Spec(known_args.tmp_path, allowed_mem=100000)
    executor = BeamDagExecutor()

    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    c = xp.add(a, b)
    res = c.compute(executor=executor, options=beam_options)
    print(res)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
