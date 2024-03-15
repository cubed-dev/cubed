import os

import modal

from cubed.runtime.utils import execute_with_stats

RUNTIME_MEMORY_MIB = 2000

stub = modal.Stub("cubed-stub")

requirements_file = os.getenv("CUBED_MODAL_REQUIREMENTS_FILE")

if requirements_file:
    image = modal.Image.debian_slim().pip_install_from_requirements(requirements_file)
    aws_image = image
    gcp_image = image
else:
    aws_image = modal.Image.debian_slim().pip_install(
        [
            "array-api-compat",
            "donfig",
            "fsspec",
            "mypy_extensions",  # for rechunker
            "networkx",
            "pytest-mock",  # TODO: only needed for tests
            "s3fs",
            "tenacity",
            "toolz",
            "zarr",
        ]
    )
    gcp_image = modal.Image.debian_slim().pip_install(
        [
            "array-api-compat",
            "donfig",
            "fsspec",
            "mypy_extensions",  # for rechunker
            "networkx",
            "pytest-mock",  # TODO: only needed for tests
            "gcsfs",
            "tenacity",
            "toolz",
            "zarr",
        ]
    )


def check_runtime_memory(spec):
    allowed_mem = spec.allowed_mem if spec is not None else None
    runtime_memory = RUNTIME_MEMORY_MIB * 1024 * 1024
    if allowed_mem is not None:
        if runtime_memory < allowed_mem:
            raise ValueError(
                f"Runtime memory ({runtime_memory}) is less than allowed_mem ({allowed_mem})"
            )


@stub.function(
    image=aws_image,
    secrets=[modal.Secret.from_name("my-aws-secret")],
    memory=RUNTIME_MEMORY_MIB,
    retries=2,
    cloud="aws",
)
def run_remotely(input, func=None, config=None, name=None, compute_id=None):
    print(f"running remotely on {input} in {os.getenv('MODAL_REGION')}")
    # note we can't use the execution_stat decorator since it doesn't work with modal decorators
    result, stats = execute_with_stats(func, input, config=config)
    return result, stats


# For GCP we need to use a class so we can set up credentials by hooking into the container lifecycle
@stub.cls(
    image=gcp_image,
    secrets=[modal.Secret.from_name("my-googlecloud-secret")],
    memory=RUNTIME_MEMORY_MIB,
    retries=2,
    cloud="gcp",
)
class Container:
    @modal.enter()
    def set_up_credentials(self):
        json = os.environ["SERVICE_ACCOUNT_JSON"]
        path = os.path.abspath("application_credentials.json")
        with open(path, "w") as f:
            f.write(json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

    @modal.method()
    def run_remotely(self, input, func=None, config=None, name=None, compute_id=None):
        print(f"running remotely on {input} in {os.getenv('MODAL_REGION')}")
        # note we can't use the execution_stat decorator since it doesn't work with modal decorators
        result, stats = execute_with_stats(func, input, config=config)
        return result, stats
