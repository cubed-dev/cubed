# Examples running Cubed on Modal

## Pre-requisites

1. A [Modal account](https://modal.com/)
2. A Google Cloud Platform (GCP) account (for Google Cloud storage)

## Set up

1. Add a new [Modal secret](https://modal.com/secrets), by following the Google Cloud wizard. This will prompt you to fill in values for `SERVICE_ACCOUNT_JSON` (it has instructions on how to create it, make sure you add the "Storage Admin" role). Call the secret `my-googlecloud-secret`.
2. Create a new GCS bucket (called `cubed-<username>-temp`, for example) in the `us-east-1` region. This will be used for intermediate data.
3. Install a Python environment by running the following from this directory:

```shell
conda create --name cubed-modal-gcp-examples -y python=3.8
conda activate cubed-modal-gcp-examples
pip install 'cubed[modal-gcp]'
export CUBED_MODAL_REQUIREMENTS_FILE=$(pwd)/requirements.txt
```

## Examples

Start with the simplest example:

```shell
python modal-add-asarray.py "gs://cubed-modal-$USER-temp"
```

If successful it should print a 4x4 matrix.

Run the other examples in a similar way

```shell
python modal-add-random.py "gs://cubed-modal-$USER-temp"
```

and

```shell
python modal-matmul-random.py "gs://cubed-modal-$USER-temp"
```

These will take longer to run as they operate on more data.

The last two examples use `TimelineVisualizationCallback` which produce a plot showing the timeline of events in the task lifecycle.
The plots are `png` files and are written in the `plots` directory with a timestamp. Open the latest one with

```shell
open plots/$(ls plots | tail -1)
```
