# Running the examples on Modal

### Pre-requisites

1. A [Modal account](https://modal.com/)
2. An AWS account (for S3 storage)

### Set up

1. Add a new [Modal secret](https://modal.com/secrets), by following the AWS wizard. This will prompt you to fill in values for `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Call the secret `my-aws-secret`.
2. Create a new S3 bucket (called `cubed-<username>-temp`, for example) in the `us-east-1` region. This will be used for intermediate data.
3. Install a Python dev environment by running the following from the top-level directory:

```shell
conda create --name cubed-modal python=3.8
conda activate cubed-modal
pip install -r requirements-modal.txt
pip install -e .
```

4. Install Modal via `pip` using the command on your home page when logged in to Modal: https://modal.com/home.

### Examples

Start with the simplest example:

```shell
python examples/modal-add-asarray.py "s3://cubed-modal-$USER-temp"
```

If successful it should print a 4x4 matrix.

Run the other examples in a similar way

```shell
python examples/modal-add-random.py "s3://cubed-modal-$USER-temp"
```

and

```shell
python examples/modal-matmul-random.py "s3://cubed-modal-$USER-temp"
```

These will take longer to run as they operate on more data.
