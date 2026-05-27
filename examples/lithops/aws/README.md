# Examples running Cubed on Lithops (AWS Lambda, S3)

## Pre-requisites

1. An AWS account, with Lambda and S3 enabled
2. A deployed Cubed Lithops runtime (see https://github.com/cubed-dev/cubed-lithops-runtime-builder-template)

## Set up

1. Install a Python environment with the basic package requirements:

```shell
conda create --name cubed-lithops-aws-examples -y python=3.12
conda activate cubed-lithops-aws-examples
pip install 'cubed[diagnostics]' 'lithops[aws]' obstore
```

2. Copy `.lithops/config` from your runtime builder repository to `~/.lithops/config`:

```shell
cp /path/to/cubed-lithops-runtime-builder/.lithops/config ~/.lithops/config
```

AWS credentials are read automatically from your [AWS CLI configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) — no credentials need to be added to the config file.

3. Create an S3 bucket for Cubed's intermediate zarr data in the same region as your Lambda function. The `cubed.yaml` config uses `$USER` to name the bucket automatically, so just run:

```shell
aws s3 mb s3://cubed-$USER-temp --region us-east-1
```

4. Set AWS environment variables for obstore:

```shell
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
```

## Running

Before running the examples, first change to the top-level examples directory (`cd ../..`) and type

```shell
export CUBED_CONFIG=$(pwd)/lithops/aws
```

Then you can run the examples in the [docs](https://cubed-dev.github.io/cubed/examples/index.html).

## Cleaning up

To rebuild the runtime after updating dependencies, push a change to your runtime builder repository — the CI pipeline rebuilds and redeploys automatically.

To remove all Lithops resources (except config files):

```shell
lithops clean -b aws_lambda
```
