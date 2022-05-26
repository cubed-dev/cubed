from apache_beam.options.pipeline_options import PipelineOptions

import cubed as xp
from cubed.runtime.executors.beam import BeamPipelineExecutor

if __name__ == "__main__":
    tmp_path = "gs://barry-zarr-test/cdtest"
    spec = xp.Spec(tmp_path, max_mem=100000)
    executor = BeamPipelineExecutor()

    beam_options = PipelineOptions(
        runner="DataflowRunner",
        project="tom-white",
        region="us-central1",
        temp_location="gs://barry-zarr-test/cdtest/tmp",
        dataflow_service_options=["use_runner_v2"],
        experiments=["use_runner_v2"],
        # Uncomment setup_file to install this project on a dataflow container
        # setup_file="/Users/tom/projects-workspace/barry/setup.py",
        # Uncomment prebuild_sdk_container_engine and docker_registry_push_url to prebuild a container
        # prebuild_sdk_container_engine="cloud_build",
        # docker_registry_push_url="us-docker.pkg.dev/tom-white/barry-test",
        # Uncomment sdk_container_image to use the latest prebuilt container (comment out setup_file in this case)
        sdk_container_image="us-docker.pkg.dev/tom-white/barry-test/beam_python_prebuilt_sdk:cd923a45-104f-4203-8b1a-29ab4974c1f2",
    )

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
    c.visualize("dataflow-add-asarray")
    res = c.compute(executor=executor, options=beam_options)
    print(res)
