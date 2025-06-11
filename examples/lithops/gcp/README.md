# Examples running Cubed on Lithops (Google Cloud Functions, GCS)

## Pre-requisites

1. A [GCP](https://cloud.google.com/) account, with [Functions](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html#installation) and [Storage](https://lithops-cloud.github.io/docs/source/storage_config/gcp_storage.html#installation) enabled.

## Set up

1. Install a Python environment with the basic package requirements:

```shell
# from this directory
conda create --name cubed-lithops-gcp-examples -y python=3.11
conda activate cubed-lithops-gcp-examples
pip install 'cubed[lithops-gcp]'
```

2. Configure Lithops with a [Google Cloud Functions compute backend](https://lithops-cloud.github.io/docs/source/compute_config/gcp_functions.html#configuration), and a [Google Cloud Storage backend](https://lithops-cloud.github.io/docs/source/storage_config/gcp_storage.html#configuration).
   - Note: it may be useful to put the configuration in a different place to the default (e.g. `~/.lithops/config.gcp`), and then call `export LITHOPS_CONFIG_FILE=~/.lithops/config.gcp`
   - Although optional, it is convenient to [configure Lithops logging](https://lithops-cloud.github.io/docs/source/configuration.html) by setting `log_filename` (to `lithops.log`, for example), so that messages are sent to a file, rather than the console.
3. Give permissions to access your GCP buckets by calling `export GOOGLE_APPLICATION_CREDENTIALS=<path-to-json>`, where `<path-to-json>` is the full path to the `.json` key file you downloaded when configuring GCP storage for Lithops in the previous step.
4. Create a new GCP bucket (called `cubed-<username>-temp`, for example) in the same region you chose when configuring Google Cloud Functions and Storage for Lithops. This will be used for intermediate zarr data. Note that this is different to the bucket created when configuring Lithops, which just stores configuration data.
5. Build a Lithops runtime image for Cubed (this is similar to building a Docker image like you need for AWS Lambda, but is simpler and faster as it only requires a `requirements.txt` file).

```shell
lithops runtime build -f requirements.txt cubed-runtime -b gcp_functions
lithops runtime deploy -b gcp_functions --memory 2048 cubed-runtime # optional, will be done automatically on first use
```

6. Set file descriptor limit. Different systems have different limits, so this step may be needed to run the larger examples. You can check what the limit is on your system with `ulimit -n`. The following command will set the limit to 1024 for the current session.

```shell
ulimit -n 1024
```

## Running

Before running the examples, first change to the top-level examples directory (`cd ../..`) and type

```shell
export CUBED_CONFIG=$(pwd)/lithops/gcp
```

Then you can run the examples in the [docs](https://cubed-dev.github.io/cubed/examples/index.html).

## Cleaning up

If you want to rebuild the Lithops runtime image you can delete the existing one by running

```shell
lithops runtime delete -b gcp_functions -d cubed-runtime
```

Or you can remove everything (except config files) with

```shell
lithops clean -b gcp_functions
```
