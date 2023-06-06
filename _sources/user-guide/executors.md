# Executors

Cubed arrays are backed by Zarr arrays, and every chunk in the Zarr array is computed by a task running on a worker, which may be a local or remote process.

Cubed provides a variety of executors for running the tasks in a computation, which are discussed below. Executors are also sometimes referred to as runtimes.

## Local Python executor

If you don't specify an executor then the local in-process Python executor is used. This is a very simple, single-threaded executor (called {py:class}`PythonDagExecutor <cubed.runtime.executors.python.PythonDagExecutor>`) that is intended for testing on small amounts of data before running larger computations using a cloud service.

## Which cloud service should I use?

[**Modal**](https://modal.com/) is the easiest to get started with because it handles building a runtime environment for you automatically (note that it requires that you [sign up](https://modal.com/signup) for a free account).
It has been tested with ~300 workers and works with AWS and GCP.

[**Lithops**](https://lithops-cloud.github.io/) requires slightly more work to get started since you have to build a runtime environment first.
Lithops has support for many serverless services on various cloud providers, but has so far been tested on two:
- **AWS Lambda** requires building a Docker container first, and has been tested with ~1000 workers.
- **Google Cloud Functions** only requires building a Lithops runtime, which can be created from a pip-style `requirements.txt` without Docker. It has been tested with ~1000 workers.

[**Google Cloud Dataflow**](https://cloud.google.com/dataflow) is relatively straightforward to get started with. It has the highest overhead for worker startup (minutes compared to seconds for Modal or Lithops), and although it has only been tested with ~20 workers, it is the most mature service and therefore should be reliable for much larger computations.

## Specifying an executor

An executor may be specified as a part of the {py:class}`Spec <cubed.Spec>`:

```python
import cubed
from cubed.runtime.executors.modal_async import AsyncModalDagExecutor

spec = cubed.Spec(
    work_dir="s3://cubed-tomwhite-temp",
    allowed_mem="2GB",
    executor=AsyncModalDagExecutor()
)
```

Alternatively an executor may be specified when {py:func}`compute() <cubed.compute>` is called. The [examples](https://github.com/tomwhite/cubed/tree/main/examples/README.md) show this in more detail for all of the cloud services described above.
