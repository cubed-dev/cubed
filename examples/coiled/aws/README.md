# Examples running Cubed on Coiled

## Pre-requisites

1. A [Coiled account](https://coiled.io/)
2. An AWS account (for S3 storage)

## Set up

1. Save your aws credentials in a ``~/.aws/credentials`` file locally, following [Coiled's instructions on accessing remote data](https://docs.coiled.io/user_guide/remote-data-access.html).
2. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the same region as your Coiled account (e.g. `us-east-1`). This will be used for intermediate data.
3. Install a Python environment with the coiled package in it by running the following from this directory:

```shell
conda create --name cubed-coiled-examples -y python=3.8
conda activate cubed-coiled-examples
pip install 'cubed[coiled]'
```

## Examples

Start with the simplest example:

```shell
python coiled-add-asarray.py "s3://cubed-$USER-temp"
```

If successful it should print a 4x4 matrix.
