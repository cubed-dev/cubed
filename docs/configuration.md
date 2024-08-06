# Configuration

Cubed uses [Donfig](https://donfig.readthedocs.io/en/latest/) for managing configuration of things like the directory used for temporary files, or for setting executor properties.

This page covers how to specify configuration properties, and a reference with all the configuration options that you can use in Cubed.

## Specification

There are three main ways of specifying configuration in Cubed:
1. by instantiating a `Spec` object,
2. by using a YAML file and setting the `CUBED_CONFIG` environment variable, or
3. by setting environment variables for individual properties.

We look at each in turn.

### `Spec` object

This is how you configure Cubed directly from within a Python program - by instantiating a {py:class}`Spec <cubed.Spec>` object:

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

### YAML file

A YAML file is a good way to encapsulate the configuration in a single file that lives outside the Python program.
It's a useful way to package up the settings for running using a particular executor, so it can be reused.
The Cubed [examples](https://github.com/cubed-dev/cubed/blob/main/examples/README.md) use YAML files for this reason.

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
| `allowed_mem`      | `200MB`           | The total memory available to a worker for running a task. This includes any `reserved_mem` that has been set.                          |
| `reserved_mem`     | `100MB`           | The memory reserved on a worker for non-data use when running a task                                                                    |
| `executor_name`    | `single-threaded` | The executor for running computations. One of `single-threaded`, `threads`, `processes`, `beam`, `coiled`, `dask`, `lithops`, `modal`.  |
| `executor_options` | `None`            | Options to pass to the executor on construction. See below for possible options for each executor.                                      |


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
| `use_backups`                | `True`  | Whether to use backup tasks for mitigating stragglers.                                             |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. The default is not to batch.                |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                          |
| `max_workers`                | `None`  | The maximum number of workers to use in the `ThreadPoolExecutor`. Defaults to number of CPU cores. |


#### `processes`

| Property                     | Default | Description                                                                                                                                |
|------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `use_backups`                | `True`  | Whether to use backup tasks for mitigating stragglers.                                                                                     |
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
| `use_backups`                | `True`  | Whether to use backup tasks for mitigating stragglers.                                                                          |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. The default is not to batch.                                             |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                                                       |
| `compute_kwargs`             | `None`  | Keyword arguments to pass to Dask's [`distributed.Client`](https://distributed.dask.org/en/latest/api.html#client) constructor. |

Note that there is currently no way to set a timeout for the Dask executor.

#### `lithops`

| Property                     | Default | Description                                                                                                                                                                                                                                                                                                                                                       |
|------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `retries`                    | 2       | The number of times to retry a task if it fails.                                                                                                                                                                                                                                                                                                                  |
| `timeout`                    | `None`  | Tasks that take longer than the timeout will be automatically killed and retried. Defaults to the timeout specified when [deploying the lithops runtime image](https://lithops-cloud.github.io/docs/source/cli.html#lithops-runtime-deploy-runtime-name). This is 180 seconds in the [examples](https://github.com/cubed-dev/cubed/blob/main/examples/README.md). |
| `use_backups`                | `True`  | Whether to use backup tasks for mitigating stragglers.                                                                                                                                                                                                                                                                                                            |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                                                                                                                                                                                                                                                                                                         |
| Other properties             | N/A     | Other properties will be passed as keyword arguments to the [`lithops.executors.FunctionExecutor`](https://lithops-cloud.github.io/docs/source/api_futures.html#lithops.executors.FunctionExecutor) constructor.                                                                                                                                                  |

Note that `batch_size` is not currently supported for Lithops.

#### `modal`

| Property                     | Default | Description                                                                         |
|------------------------------|---------|-------------------------------------------------------------------------------------|
| `cloud`                      | `aws`   | The cloud to run on. One of `aws` or `gcp`.                                         |
| `use_backups`                | `True`  | Whether to use backup tasks for mitigating stragglers.                              |
| `batch_size`                 | `None`  | Number of input tasks to submit to be run in parallel. The default is not to batch. |
| `compute_arrays_in_parallel` | `False` | Whether arrays are computed one at a time or in parallel.                           |

Currently the Modal executor in Cubed uses a hard-coded value of 2 for retries and 300 seconds for timeouts, neither of which can be changed through configuration.
