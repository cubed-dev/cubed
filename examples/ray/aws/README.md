# Examples running Cubed on Ray (AWS)

## Pre-requisites

1. An AWS account

## Start Ray Cluster

Use Ray's Cluster Launcher to launch a Ray cluster on EC2, following https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html.

1. Install a Python environment by running the following from this directory:

```shell
conda create --name ray-aws-cluster -y python=3.12
conda activate ray-aws-cluster
pip install "ray[default]" boto3
```

2. Start the Ray cluster (after possibly adjusting `config.yaml` settings for region and number of workers)

```shell
ray up -y config.yaml
ray dashboard config.yaml # (optional) useful way to port forward to view dashboard on http://localhost:8265/
```

## Set up

1. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the same region you started the Ray cluster in. This will be used for intermediate zarr data.

2. Attach to the head node

```shell
ray attach config.yaml
```

3. Clone the Cubed repository

```shell
git clone https://github.com/cubed-dev/cubed
cd cubed/examples
pip install "cubed[diagnostics]" s3fs
export CUBED_CONFIG=$(pwd)/ray/aws
export USER=...
```

4. Set environment variables for AWS credentials so they are available on the workers for S3 access.

```shell
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Note that there is another way to do this described in the Ray documentation: https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html#accessing-s3.

## Examples

Now run the examples in the [docs](https://cubed-dev.github.io/cubed/examples/index.html).

## Shutdown Ray Cluster

Don't forget to shutdown the Ray cluster:

```shell
ray down -y config.yaml
```
