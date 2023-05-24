# Examples running Cubed on Lithops (AWS Lambda, S3)

## Pre-requisites

1. An AWS account, with Lambda and S3 enabled
2. [Docker Desktop](https://docs.docker.com/get-docker/)

## Set up

1. Install a Python environment with the basic package requirements:

```shell
conda create --name cubed-lithops-examples -y python=3.8
conda activate cubed-lithops-examples
pip install -r requirements.txt
```

2. Configure Lithops with an [AWS Lambda compute backend](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html), and an [AWS S3 storage backend](https://lithops-cloud.github.io/docs/source/storage_config/aws_s3.html).
   - Note: it may be useful to put the configuration in a different place to the default (e.g. `~/.lithops/config.aws`), and then call `export LITHOPS_CONFIG_FILE=~/.lithops/config.aws`)
3. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the same region you chose when configuring Lamda and S3 for Lithops. This will be used for intermediate data. Note that this is different to the bucket created when configuring Lithops.
4. Build a Lithops runtime image for Cubed (this will build and upload a Docker image, which can take a while, although it only needs to be done once).
   - Note: if you are building on an arm64 machine (e.g. Apple Silicon) then make sure that your Lithops config file contains `architecture: arm64` under the `aws_lambda` section.

```shell
lithops runtime build -b aws_lambda -f docker/Dockerfile_aws_lambda cubed-runtime
lithops runtime deploy -b aws_lambda --memory 2000 --timeout 180 cubed-runtime # optional, will be done automatically on first use
```

## Running

Start with the simplest example:

```shell
python lithops-add-asarray.py "s3://cubed-$USER-temp" cubed-runtime
```

If successful it should print a 4x4 matrix.

Run the other example in a similar way

```shell
python lithops-add-random.py "s3://cubed-$USER-temp" cubed-runtime
```
