from apache_beam.options.pipeline_options import PipelineOptions

import cubed as xp
import cubed.random
from cubed.rechunker_extensions.executors.beam import BeamDagExecutor

if __name__ == "__main__":
    tmp_path = "gs://barry-zarr-test/cdtest"
    spec = xp.Spec(tmp_path, max_mem=1_000_000_000)
    executor = BeamDagExecutor()

    beam_options = PipelineOptions(
        runner="DataflowRunner",
        project="tom-white",
        region="us-central1",
        temp_location="gs://barry-zarr-test/cdtest/tmp",
        dataflow_service_options=["use_runner_v2"],
        autoscaling_algorithm="NONE",
        num_workers=4,
        experiments=["use_runner_v2"],
        sdk_container_image="us-docker.pkg.dev/tom-white/barry-test/beam_python_prebuilt_sdk:cd923a45-104f-4203-8b1a-29ab4974c1f2",
    )

    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    c.visualize("dataflow-add-random")
    c.compute(return_stored=False, executor=executor, options=beam_options)
