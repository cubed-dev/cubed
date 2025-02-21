# Configuration

Cubed uses [Donfig](https://donfig.readthedocs.io/en/latest/) for managing configuration of things like the directory used for temporary files, or for setting executor properties.

This page covers how to specify configuration properties, and a reference with all the configuration options that you can use in Cubed.

## Specification

There are four main ways of specifying configuration in Cubed:
1. by instantiating a `Spec` object,
2. by setting values on the `config` object in the `cubed` namespace,
3. by using a YAML file and setting the `CUBED_CONFIG` environment variable, or
4. by setting environment variables for individual properties.

We look at each in turn.

### `Spec` object

This is the most direct way to configure Cubed directly from within a Python program - by instantiating a {py:class}`Spec <cubed.Spec>` object:

```python
import cubed

spec = cubed.Spec(
    work_dir="s3://cubed-tomwhite-temp",
    allowed_mem="2GB",
    executor_name="lithops",
    executor_options=dict(use_backups=False, runtime="cubed-runtime", runtime_memory=2000)
)
```

The `spec` instance is then passed to array creation functions as follows:

```python
import cubed.array_api as xp

a = cubed.random.random((50000, 50000), chunks=(5000, 5000), spec=spec)
b = cubed.random.random((50000, 50000), chunks=(5000, 5000), spec=spec)
c = xp.add(a, b)
```

All arrays in any given computation must share the same `spec` instance.

### `cubed.config` object

This way allows you to set configuration globally, or using a context manager for a block of code.

The following sets the configuration globally:

```python
from cubed import config
config.set({
  "spec.work_dir": "s3://cubed-tomwhite-temp",
  "spec.allowed_mem": "2GB",
  "spec.executor_name": "lithops",
  "spec.executor_options.use_backups": False,
  "spec.executor_options.runtime": "cubed-runtime",
  "spec.executor_options.runtime_memory": 2000,
})
```

There is no need to pass a `spec` object to array creation functions when setting configuration this way.

Use a `with` statement to limit the configuration overrides to a code block:

```python
from cubed import config
import cubed.array_api as xp

with config.set({"spec.executor_name": "single-threaded"}):
  a = cubed.random.random((50000, 50000), chunks=(5000, 5000))
  b = cubed.random.random((50000, 50000), chunks=(5000, 5000))
  c = xp.add(a, b)
```

### YAML file

A YAML file is a good way to encapsulate the configuration in a single file that lives outside the Python program.
It's a useful way to package up the settings for running using a particular executor, so it can be reused.
The Cubed [examples](examples/index.md) use YAML files for this reason.

```yaml
spec:
  work_dir: "s3://cubed-$USER-temp"
  allowed_mem: "2GB"
  executor_name: "lithops"
  executor_options:
    use_backups: False
    runtime: "cubed-runtime"
    runtime_memory: 2000
```

Note that YAML files can use environment variables - in this example `$USER` will be expanded appropriately.

To use the YAML file, set the `CUBED_CONFIG` environment variable to the file path before invoking your Python program:

```shell
CUBED_CONFIG=/path/to/cubed.yaml python ...
```

Donfig will actually look for YAML files in a variety of locations, see the [Donfig docs](https://donfig.readthedocs.io/en/latest/configuration.html#yaml-files) for details.

### Environment variables

You can also set [Donfig-style environment variables](https://donfig.readthedocs.io/en/latest/configuration.html#environment-variables) to set individual properties. Notice how double underscores are used to indicate nesting.

```shell
export CUBED_SPEC__WORK_DIR='s3://cubed-$USER-temp'
export CUBED_SPEC__ALLOWED_MEM=2GB
export CUBED_SPEC__EXECUTOR_NAME=lithops
export CUBED_SPEC__EXECUTOR_OPTIONS__USE_BACKUPS=False
export CUBED_SPEC__EXECUTOR_OPTIONS__RUNTIME=cubed-runtime
export CUBED_SPEC__EXECUTOR_OPTIONS__RUNTIME_MEMORY=2000
```

This can be handy if you only have a couple of properties to set:

```shell
CUBED_SPEC__ALLOWED_MEM=2GB CUBED_SPEC__EXECUTOR_NAME=processes python ...
```

## Reference

### Spec options

These properties can be passed directly to the {py:class}`Spec <cubed.Spec>` constructor. Or, equivalently, they are directly under `spec` in a YAML file.

| Property           | Default           | Description                                                                                                                             |
|--------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `work_dir`         | `None`            | The directory path (specified as an fsspec URL) used for storing intermediate data. If not set, the user's temporary directory is used. |
| `allowed_mem`      | `"2GB"`           | The total memory available to a worker for running a task. This includes any `reserved_mem` that has been set.                          |
| `reserved_mem`     | `"100MB"`         | The memory reserved on a worker for non-data use when running a task                                                                    |
| `executor_name`    | `"threads"`       | The executor for running computations. One of `"single-threaded"`, `"threads"`, `"processes"`, `"beam"`, `"coiled"`, `"dask"`, `"lithops"`, `"modal"`.  |
| `executor_options` | `None`            | Options to pass to the executor on construction. See below for possible options for each executor.                                      |
| `zarr_compressor`  | `"default"`| The compressor used by Zarr for intermediate data. If not specified, or set to `"default"`, Zarr will use the default Blosc compressor. If set to `None`, compression is disabled, which can be a good option when using local storage. Use a dictionary (or nested YAML) to configure arbitrary compression using Numcodecs. |

### Executor options

Different executors support different options. They are listed here for each executor.

These properties are keys in the `executor_options` passed to the {py:class}`Spec <cubed.Spec>` constructor. Or, equivalently, they are directly under `spec.executor_options` in a YAML file.


#### `single-threaded`

The `single-threaded` executor is a simple executor mainly used for testing. It doesn't support any configuration options
since it is deliberately designed not to have anything except the most basic features.

#### `threads`

| Property                     | Default | Description                                                                                        |
|------------------------------|---------|----------------------------------------------------------------------------------------------------|
| `retries`                    | 2       | The number of times to retry a task if it fails.                                                   |
| `use_backups`                |         | Whether to use backup tasks for mitigating stragglers. Defaults to `True` only if `work_dir` is a filesystem supporting atomic writes (currently a cloud store like S3 or GCS). |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. The default is not to batch.                |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                          |
| `max_workers`                | `None`  | The maximum number of workers to use in the `ThreadPoolExecutor`. Defaults to number of CPU cores. |


#### `processes`

| Property                     | Default | Description                                                                                                                                |
|------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `use_backups`                |         | Whether to use backup tasks for mitigating stragglers. Defaults to `True` only if `work_dir` is a filesystem supporting atomic writes (currently a cloud store like S3 or GCS). |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. `None` means don't batch.                                                           |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                                                                  |
| `max_workers`                | `None`  | The maximum number of workers to use in the `ProcessPoolExecutor`. Defaults to number of CPU cores.                                        |
| `max_tasks_per_child`        | `None`  | The number of tasks to run in each child process. See the Python documentation for `concurrent.futures.ProcessPoolExecutor`. (Python 3.11) |

Note that `retries` is not currently supported for the `processes` executor.

#### `beam`

The `beam` executor doesn't currently expose any configuration options.
When running on Google Cloud Dataflow, [four retry attempts](https://cloud.google.com/dataflow/docs/pipeline-lifecycle#error_and_exception_handling) are made for failing tasks.

#### `coiled`

| Property        | Default | Description                                                                                             |
|-----------------|---------|---------------------------------------------------------------------------------------------------------|
| `coiled_kwargs` | `None`  | Keyword arguments to pass to [`coiled.function`](https://docs.coiled.io/user_guide/functions.html#api). |

Note that there is currently no way to set retries or a timeout for the Coiled executor.

#### `dask`

| Property                     | Default | Description                                                                                                                     |
|------------------------------|---------|---------------------------------------------------------------------------------------------------------------------------------|
| `retries`                    | 2       | The number of times to retry a task if it fails.                                                                                |
| `use_backups`                |         | Whether to use backup tasks for mitigating stragglers. Defaults to `True` only if `work_dir` is a filesystem supporting atomic writes (currently a cloud store like S3 or GCS). |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. The default is not to batch.                                             |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                                                       |
| `compute_kwargs`             | `None`  | Keyword arguments to pass to Dask's [`distributed.Client`](https://distributed.dask.org/en/latest/api.html#client) constructor. |

Note that there is currently no way to set a timeout for the Dask executor.

#### `lithops`

| Property                     | Default | Description                                                                                                                                                                                                                                                                                                                                                       |
|------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `retries`                    | 2       | The number of times to retry a task if it fails.                                                                                                                                                                                                                                                                                                                  |
| `timeout`                    | `None`  | Tasks that take longer than the timeout will be automatically killed and retried. Defaults to the timeout specified when [deploying the lithops runtime image](https://lithops-cloud.github.io/docs/source/cli.html#lithops-runtime-deploy-runtime-name). This is 180 seconds in the [examples](https://github.com/cubed-dev/cubed/blob/main/examples/lithops/aws/README.md). |
| `use_backups`                |         | Whether to use backup tasks for mitigating stragglers. Defaults to `True` only if `work_dir` is a filesystem supporting atomic writes (currently a cloud store like S3 or GCS). |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                                                                                                                                                                                                                                                                                         |
| Other properties             | N/A     | Other properties will be passed as keyword arguments to the [`lithops.executors.FunctionExecutor`](https://lithops-cloud.github.io/docs/source/api_futures.html#lithops.executors.FunctionExecutor) constructor.                                                                                                                                                  |

Note that `batch_size` is not currently supported for Lithops.

#### `modal`

| Property                     | Default | Description                                                                         |
|------------------------------|---------|-------------------------------------------------------------------------------------|
| `cloud`                      | `"aws"` | The cloud to run on. One of `"aws"` or `"gcp"`.                                     |
| `region`                     | N/A     | The cloud region to run in. This must be set to match the region of your cloud store to avoid data transfer fees. See Modal's [Region selection](https://modal.com/docs/guide/region-selection) page for possible values. |
| `retries`                    | 2       | The number of times to retry a task if it fails.                                    |
| `timeout`                    | 180     | Tasks that take longer than the timeout will be automatically killed and retried.   |
| `enable_output`              | False   | Print Modal output to stdout and stderr things for debuggging.                      |
| `use_backups`                | `True`  | Whether to use backup tasks for mitigating stragglers.                              |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. The default is not to batch. |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                           |

## Debugging

You can use Donfig's `pprint` method if you want to check which configuration settings are in effect when you code is run:

```python
from cubed import config
config.pprint()
```
