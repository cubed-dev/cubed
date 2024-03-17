# Executors

Cubed arrays are backed by Zarr arrays, and every chunk in the Zarr array is computed by a task running on a worker, which may be a local or remote process.

Cubed provides a variety of executors for running the tasks in a computation, which are discussed below. Executors are also sometimes referred to as runtimes.

## Local Python executor

If you don't specify an executor then the local in-process Python executor is used. This is a very simple, single-threaded executor (called {py:class}`PythonDagExecutor <cubed.runtime.executors.python.PythonDagExecutor>`) that is intended for testing on small amounts of data before running larger computations using a cloud service.

## Which cloud service executor should I use?

[**Lithops**](https://lithops-cloud.github.io/) is the executor we recommend for most users, since it has had the most testing so far (~1000 workers).
If your data is in Amazon S3 then use Lithops with AWS Lambda, and if it's in GCS use Lithops with Google Cloud Functions. You have to build a runtime environment as a part of the setting up process.

[**Modal**](https://modal.com/) is very easy to get started with because it handles building a runtime environment for you automatically (note that it requires that you [sign up](https://modal.com/signup) for a free account). **At the time of writing, Modal does not guarantee that functions run in any particular cloud region, so it is not currently recommended that you run large computations since excessive data transfer fees are likely.**

[**Coiled**](https://www.coiled.io/) is also easy to get started with ([sign up](https://cloud.coiled.io/signup)). It uses [Coiled Functions](https://docs.coiled.io/user_guide/usage/functions/index.html) and has a 1-2 minute overhead to start a cluster.

[**Google Cloud Dataflow**](https://cloud.google.com/dataflow) is relatively straightforward to get started with. It has the highest overhead for worker startup (minutes compared to seconds for Modal or Lithops), and although it has only been tested with ~20 workers, it is a mature service and therefore should be reliable for much larger computations.

## Specifying an executor

An executor may be specified as a part of the {py:class}`Spec <cubed.Spec>`:

```python
import cubed

spec = cubed.Spec(
    work_dir="s3://cubed-tomwhite-temp",
    allowed_mem="2GB",
    executor_name="modal"
)
```

A default spec may also be configured using a YAML file. The [examples](https://github.com/tomwhite/cubed/tree/main/examples/README.md) show this in more detail for all of the cloud services described above.
