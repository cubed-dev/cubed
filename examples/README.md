# Examples

## Lithops (AWS Lambda, S3)

### Pre-requisites

1. An AWS account, with Lambda and S3 enabled
2. [Docker Desktop](https://docs.docker.com/get-docker/)

### Set up

1. Install a Python dev environment as described in the top-level README.
2. Configure Lithops with an [AWS Lambda compute backend](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html), and an [AWS S3 storage backend](https://lithops-cloud.github.io/docs/source/storage_config/aws_s3.html).
   - Note: it may be useful to put the configuration in a different place to the default (e.g. `~/.lithops/config.aws`), and then call `export LITHOPS_CONFIG_FILE=~/.lithops/config.aws`)
3. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the same region you chose when configuring Lamda and S3 for Lithops. This will be used for intermediate data. Note that this is different to the bucket created when configuring Lithops.
4. Build a Lithops runtime image for Cubed (this will build and upload a Docker image, which can take a while):

```shell
lithops runtime build -b aws_lambda -f examples/docker/Dockerfile_aws_lambda cubed-runtime
lithops runtime deploy -b aws_lambda --memory 2000 --timeout 180 cubed-runtime # optional, will be done automatically on first use
```

### Running

Start with the simplest example:

```shell
python examples/lithops-add-asarray.py "s3://cubed-$USER-temp" cubed-runtime
```

If successful it should print a 4x4 matrix.

Run the other example in a similar way

```shell
python examples/lithops-add-random.py "s3://cubed-$USER-temp" cubed-runtime
```

## Apache Beam (Google Cloud Dataflow)

### Pre-requisites

1. A GCP account, with Dataflow and Storage enabled

### Set up

1. Install a Python dev environment as described in the top-level README.
2. Create a new GCS bucket (called `cubed-<username>-temp`, for example) in a given region (e.g. `us-central1`) for low latency. This will be used for intermediate data.

### Running

Run the following script to run a small computation.

- `DATAFLOW_REGION` should be the same region as the GCS bucket created above.
- `PROJECT_ID` is your GCP project ID.

```shell
DATAFLOW_REGION=...
PROJECT_ID=...
python examples/dataflow-add-asarray.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --setup_file $PWD/setup.py
```

If successful it should print a 4x4 matrix.

The other examples are run similarly. However, it's a good idea to specify the number of workers needed:

```shell
python examples/dataflow-add-random.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --autoscaling_algorithm NONE \
    --num_workers 4 \
    --setup_file $PWD/setup.py
```

For `dataflow-matmul-random.py`, try with `--num_workers 8`.

### Pre-building a container

Pre-building a custom container can help reduce the worker startup time. To do this, first enable the Google Cloud Build API and Artifact Registry services. Then:

1. Create a new repository called `cubed-runtime` in Artifact Registry. The format should be 'Docker', and the Location type 'Region', in the same region as the GCS bucket created above. Click on the new repository and copy its URL.
2. Run the following script to run a small computation. This will also [pre-build a container for later use](https://cloud.google.com/dataflow/docs/guides/using-custom-containers#prebuild).

   - `DOCKER_REGISTRY_PUSH_URL` is the registry URL copied above.

```shell
DATAFLOW_REGION=...
PROJECT_ID=...
DOCKER_REGISTRY_PUSH_URL=...
python examples/dataflow-add-asarray.py --tmp_path gs://cubed-$USER-temp \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $DATAFLOW_REGION \
    --temp_location gs://cubed-$USER-temp \
    --dataflow_service_options use_runner_v2 \
    --experiments use_runner_v2 \
    --setup_file $PWD/setup.py \
    --prebuild_sdk_container_engine cloud_build \
    --docker_registry_push_url $DOCKER_REGISTRY_PUSH_URL
```

The image name is printed to the log, for example:

```
INFO:apache_beam.runners.portability.sdk_container_builder:Python SDK container built and pushed as us-central1-docker.pkg.dev/tom-white/cubed-runtime/beam_python_prebuilt_sdk:7b9fe2ea-f4d0-4725-abcf-6514c858d3f2.
```

Use this name with the `--sdk_container_image` option instead of specifying `--setup_file` in the examples of the previous section.
