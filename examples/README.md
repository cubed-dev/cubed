# Examples

## Lithops (AWS Lambda, S3)

### Pre-requisites

1. An AWS account, with Lambda and S3 enabled
2. [Docker Desktop](https://docs.docker.com/get-docker/)

### Set up

1. Install a Python dev environment as described in the top-level README.
2. Configure Lithops with an [AWS Lambda compute backend](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html), and an [AWS S3 storage backend](https://lithops-cloud.github.io/docs/source/storage_config/aws_s3.html).
3. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the same region you chose when configuring Lamda and S3 for Lithops. This will be used for intermediate data. Note that this is different to the bucket created when configuring Lithops.
4. Build a Lithops runtime image for Cubed (this will build and upload a Docker image, which can take a while):

```shell
lithops runtime build -b aws_lambda -f examples/docker/Dockerfile_aws_lambda cubed-runtime
```

### Running

Start with the simplest example:

```shell
python examples/lithops-add-asarray.py "s3://cubed-$USER-temp" cubed-runtime
```

If successful it should print a 4x4 matrix.

Run the other example in a similar way

```shell
python examples/lithops-add-random.py "s3://cubed-$USER-temp" cubed-runtime
```
