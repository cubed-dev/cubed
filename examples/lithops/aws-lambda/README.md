# Examples running Cubed on Lithops (AWS Lambda, S3)

## Pre-requisites

1. An AWS account, with Lambda and S3 enabled
2. [Docker Desktop](https://docs.docker.com/get-docker/)

## Set up

1. Install a Python environment with the basic package requirements:

```shell
conda create --name cubed-lithops-aws-examples -y python=3.9
conda activate cubed-lithops-aws-examples
pip install -r requirements.txt  # use requirements file from same directory as this readme
```

2. Configure Lithops with an [AWS Lambda compute backend](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html), and an [AWS S3 storage backend](https://lithops-cloud.github.io/docs/source/storage_config/aws_s3.html).
   - Note: it may be useful to put the configuration in a different place to the default (e.g. `~/.lithops/config.aws`), and then call `export LITHOPS_CONFIG_FILE=~/.lithops/config.aws`
   - Although optional, it is convenient to [configure Lithops logging](https://lithops-cloud.github.io/docs/source/configuration.html) by setting `log_filename` (to `lithops.log`, for example), so that messages are sent to a file, rather than the console.
3. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the same region you chose when configuring Lambda and S3 for Lithops. This will be used for intermediate zarr data. Note that this is different to the bucket created when configuring Lithops, which just stores configuration data.
4. Build a Lithops runtime image for Cubed (this will build and upload a Docker image, which can take a while, although it only needs to be done once).
   - Note: if you are building on an arm64 machine (e.g. Apple Silicon) then make sure that your Lithops config file contains `architecture: arm64` under the `aws_lambda` section.

```shell
lithops runtime build -b aws_lambda -f docker/Dockerfile_aws_lambda cubed-runtime
lithops runtime deploy -b aws_lambda --memory 2000 --timeout 180 cubed-runtime # optional, will be done automatically on first use
```

5. Set file descriptor limit. Different systems have different limits, so this step may be needed to run the larger examples. You can check what the limit is on your system with `ulimit -n`. The following command will set the limit to 1024 for the current session.

```shell
ulimit -n 1024
```

## Running

Before running the examples, first change to the top-level examples directory (`cd ../..`) and type

```shell
export CUBED_CONFIG=$(pwd)/lithops/aws-lambda
```

Then you can run the examples described [there](../../README.md).

## Cleaning up

If you want to rebuild the Lithops runtime image you can delete the existing one by running

```shell
lithops runtime delete -b aws_lambda -d cubed-runtime
```

Or you can remove everything (except config files) with

```shell
lithops clean -b aws_lambda
```
