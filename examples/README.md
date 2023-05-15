# Examples

## Which cloud service should I use?

**Modal** is the easiest to get started with (if you have an account - it's currently in private beta). It has been tested with ~50 workers.

**Lithops** requires more work to get started since you have to build a docker container first. It has been tested with hundreds of workers, but only on AWS Lambda, although Lithops has support for many more serverless services on various cloud providers.

**Google Cloud Dataflow** is relatively strightforward to get started with. It has the highest overhead for worker startup (minutes compared to seconds for Modal or Lithops), and although it has only been tested with ~20 workers, it is the most mature service and therefore should be reliable for much larger computations.

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

## Modal

### Pre-requisites

1. A [Modal account](https://modal.com/)
2. An AWS account (for S3 storage)

### Set up

1. Add a new [Modal secret](https://modal.com/secrets), by following the AWS wizard. This will prompt you to fill in values for `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Call the secret `my-aws-secret`.
2. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the `us-east-1` region. This will be used for intermediate data.
3. Install a Python environment by running the following from this directory:

```shell
conda create --name cubed-modal-examples -y python=3.8
conda activate cubed-modal-examples
pip install 'cubed[modal]'
export CUBED_MODAL_REQUIREMENTS_FILE=$(pwd)/requirements.txt
```

### Examples

Start with the simplest example:

```shell
python modal-add-asarray.py "s3://cubed-modal-$USER-temp"
```

If successful it should print a 4x4 matrix.

Run the other examples in a similar way

```shell
python modal-add-random.py "s3://cubed-modal-$USER-temp"
```

and

```shell
python modal-matmul-random.py "s3://cubed-modal-$USER-temp"
```

These will take longer to run as they operate on more data.

## Apache Beam (Google Cloud Dataflow)

### Pre-requisites

1. A GCP account, with Dataflow and Storage enabled

### Set up

1. Install a Python dev environment as described in the top-level README.
2. Install the dataflow dependencies with `pip install apache-beam[gcp]`
3. Create a new GCS bucket (called `cubed-<username>-temp`, for example) in a given region (e.g. `us-central1`) for low latency. This will be used for intermediate data.

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

The other examples are can be run similarly. However, it's a good idea to pre-build a custom container first.

### Pre-building a container

Pre-building a custom container can help reduce the worker startup time. It also makes it possible
to specify `--no_use_public_ips` which allows us to run larger clusters (so they are not limited by the number of in-use IP addresses quota).

To do this, first enable the Google Cloud Build API and Artifact Registry services. Then:

1. Create a new repository called `cubed-runtime` in Artifact Registry. The format should be 'Docker', and the Location type 'Region', in the same region as the GCS bucket created above. Click on the new repository and copy its URL.
2. In the top-level _setup.cfg_ file ensure that the `gcsfs` line for `install_requires` is uncommented.
3. Run the following script to run a small computation. This will also [pre-build a container for later use](https://cloud.google.com/dataflow/docs/guides/using-custom-containers#prebuild).

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

Use this name with the `--sdk_container_image` option instead of specifying `--setup_file` in the example of the previous section.

Also, in order to take advantage of `--no_use_public_ips`, you need to follow the instructions for [Enabling Private Google Access](https://cloud.google.com/vpc/docs/configure-private-google-access#enabling-pga).

Example for a larger `add` operation:

```shell
python examples/dataflow-add-random.py --tmp_path gs://cubed-$USER-temp \
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
python examples/dataflow-matmul-random.py --tmp_path gs://cubed-$USER-temp \
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
