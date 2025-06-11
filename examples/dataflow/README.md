# Examples running Cubed on Apache Beam (Google Cloud Dataflow)

## Pre-requisites

1. A GCP account, with Dataflow and Storage enabled

## Set up

1. Install a Python environment with the basic package requirements:

```shell
conda create --name cubed-dataflow-examples -y python=3.10
conda activate cubed-dataflow-examples
pip install -r requirements.txt
```
2. Create a new GCS bucket (called `cubed-<username>-temp`, for example) in a given region (e.g. `us-central1`) for low latency. This will be used for intermediate data.

## Running

Run the following script to run a small computation.

- `DATAFLOW_REGION` should be the same region as the GCS bucket created above.
- `PROJECT_ID` is your GCP project ID.

```shell
DATAFLOW_REGION=...
PROJECT_ID=...
python dataflow-add-asarray.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp/temp \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --setup_file $(pwd)/setup.py
```

Note that we use `--setup_file` rather than `--requirements_file` because the latter only installs source packages.

If successful it should print a 4x4 matrix.

The other examples are can be run similarly. However, it's a good idea to pre-build a custom container first.


### Pre-building a container

Pre-building a custom container can help reduce the worker startup time. It also makes it possible
to specify `--no_use_public_ips` which allows us to run larger clusters (so they are not limited by the number of in-use IP addresses quota).

To do this, first enable the Google Cloud Build API and Artifact Registry services. Then:

1. Create a new repository called `cubed-runtime` in Artifact Registry. The format should be 'Docker', and the Location type 'Region', in the same region as the GCS bucket created above. Click on the new repository and copy its URL.
2. Run the following script to run a small computation. This will also [pre-build a container for later use](https://cloud.google.com/dataflow/docs/guides/using-custom-containers#prebuild).

   - `DOCKER_REGISTRY_PUSH_URL` is the registry URL copied above.

```shell
DATAFLOW_REGION=...
PROJECT_ID=...
DOCKER_REGISTRY_PUSH_URL=...
python events_dfdataflow-add-asarray.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --setup_file $(pwd)/setup.py \
    --prebuild_sdk_container_engine cloud_build \
    --docker_registry_push_url $DOCKER_REGISTRY_PUSH_URL
```

The image name is printed to the log, for example:

```
INFO:apache_beam.runners.portability.sdk_container_builder:Python SDK container built and pushed as us-central1-docker.pkg.dev/tom-white/cubed-runtime/beam_python_prebuilt_sdk:7b9fe2ea-f4d0-4725-abcf-6514c858d3f2.
```

Use this name with the `--sdk_container_image` option instead of specifying `--setup_file` in the example of the previous section.

Also, in order to take advantage of `--no_use_public_ips`, you need to follow the instructions for [Enabling Private Google Access](https://cloud.google.com/vpc/docs/configure-private-google-access#enabling-pga).

Example for a larger `add` operation:

```shell
python events_dfdataflow-add-random.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp \
    --no_use_public_ips \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --autoscaling_algorithm NONE \
    --num_workers 12 \
    --sdk_container_image us-central1-docker.pkg.dev/tom-white/cubed-runtime/beam_python_prebuilt_sdk:5ea62e51-fc03-43e6-898f-c109288484fb
```

Example for `matmul`:

```shell
python events_dfdataflow-matmul-random.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp \
    --no_use_public_ips \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --autoscaling_algorithm NONE \
    --num_workers 24 \
    --sdk_container_image us-central1-docker.pkg.dev/tom-white/cubed-runtime/beam_python_prebuilt_sdk:5ea62e51-fc03-43e6-898f-c109288484fb
```
