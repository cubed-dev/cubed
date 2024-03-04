# Examples running Cubed on Modal

**Warning: Modal does not guarantee that functions run in any particular cloud region, so it is not currently recommended that you run large computations since excessive data transfer fees are likely.**

## Pre-requisites

1. A [Modal account](https://modal.com/)
2. An AWS account (for S3 storage)

## Set up

1. Add a new [Modal secret](https://modal.com/secrets), by following the AWS wizard. This will prompt you to fill in values for `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Call the secret `my-aws-secret`.
2. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the `us-east-1` region. This will be used for intermediate data.
3. Install a Python environment by running the following from this directory:

```shell
conda create --name cubed-modal-aws-examples -y python=3.11
conda activate cubed-modal-aws-examples
pip install 'cubed[modal]'
export CUBED_MODAL_REQUIREMENTS_FILE=$(pwd)/requirements.txt
```

## Examples

Before running the examples, first change to the top-level examples directory (`cd ../..`) and type

```shell
export CUBED_CONFIG=$(pwd)/modal/aws
```

Then you can run the examples described [there](../../README.md).
