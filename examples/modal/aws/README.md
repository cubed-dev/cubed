# Examples running Cubed on Modal

**Warning: Modal does not guarantee that functions run in any particular cloud region, so it is not currently recommended that you run large computations since excessive data transfer fees are possible.**

## Pre-requisites

1. A [Modal account](https://modal.com/)
2. An AWS account (for S3 storage)

## Set up

1. Add a new [Modal secret](https://modal.com/secrets), by following the AWS wizard. This will prompt you to fill in values for `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Call the secret `my-aws-secret`.
2. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the `us-east-1` region. This will be used for intermediate data.
3. Install a Python environment by running the following from this directory:

```shell
conda create --name cubed-modal-examples -y python=3.9
conda activate cubed-modal-examples
pip install 'cubed[modal]'
export CUBED_MODAL_REQUIREMENTS_FILE=$(pwd)/requirements.txt
```

## Examples

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

The last two examples use `TimelineVisualizationCallback` which produce a plot showing the timeline of events in the task lifecycle.
The plots are `png` files and are written in the `plots` directory with a timestamp. Open the latest one with

```shell
open plots/$(ls plots | tail -1)
```
