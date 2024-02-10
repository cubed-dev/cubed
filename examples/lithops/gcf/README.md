# Examples running Cubed on Lithops (Google Cloud Functions, GCS)

## Pre-requisites

1. A [GCP](https://cloud.google.com/) account, with [Functions](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html#installation) and [Storage](https://lithops-cloud.github.io/docs/source/storage_config/gcp_storage.html#installation) enabled.

## Set up

1. Install a Python environment with the basic package requirements:

```shell
# from this directory
conda create --name cubed-lithops-gcf-examples -y python=3.9
conda activate cubed-lithops-gcf-examples
pip install -r requirements.txt  # use requirements file from same directory as this readme
```

2. Configure Lithops with a [Google Cloud Functions compute backend](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html#configuration), and a [Google Cloud Storage backend](https://lithops-cloud.github.io/docs/source/storage_config/gcp_storage.html#configuration).
   - Note: it may be useful to put the configuration in a different place to the default (e.g. `~/.lithops/config.gcf`), and then call `export LITHOPS_CONFIG_FILE=~/.lithops/config.gcf`)
3. Give permissions to access your GCP buckets by calling `export GOOGLE_APPLICATION_CREDENTIALS=<path-to-json>`, where `<path-to-json>` is the full path to the `.json` key file you downloaded when configuring GCP storage for Lithops in the previous step.
4. Create a new GCP bucket (called `cubed-<username>-temp`, for example) in the same region you chose when configuring Google Cloud Functions and Storage for Lithops. This will be used for intermediate zarr data. Note that this is different to the bucket created when configuring Lithops, which just stores configuration data.
5. Build a Lithops runtime image for Cubed (this is similar to building a Docker image like you need for AWS Lambda, but is simpler and faster as it only requires a `requirements.txt` file).

```shell
lithops runtime build -f requirements.txt cubed-runtime -b gcp_functions
lithops runtime deploy -b gcp_functions --memory 2048 cubed-runtime # optional, will be done automatically on first use
```

## Running

Start with the simplest example:

```shell
python lithops-add-asarray.py "gs://cubed-$USER-temp" cubed-runtime
```

If successful it should print a 4x4 matrix.

Run the other examples in a similar way

```shell
python lithops-add-random.py "gs://cubed-$USER-temp" cubed-runtime
```

and

```shell
python lithops-matmul-random.py "gs://cubed-$USER-temp" cubed-runtime
```

These will take longer to run as they operate on more data.


The last two examples use `TimelineVisualizationCallback` which produce a plot showing the timeline of events in the task lifecycle.
The plots are `png` files and are written in the `history` directory in a directory with a timestamp. Open the latest one with

```shell
open $(ls -d history/compute-* | tail -1)/timeline.png
```

## Cleaning up

If you want to rebuild the Lithops runtime image you can delete the existing one by running

```shell
lithops runtime delete -b gcp_functions -d cubed-runtime
```

Or you can remove everything (except config files) with

```shell
lithops clean -b gcp_functions
```
