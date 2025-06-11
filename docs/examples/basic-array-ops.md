# Basic array operations

The following examples show how to run a few basic Array API operations on Cubed arrays.

## Adding two small arrays

The first example adds two small 4x4 arrays together, and is useful for checking that the runtime is working.

```{eval-rst}
.. literalinclude:: ../../examples/add-asarray.py
```

Paste the code into a file called `add-asarray.py`, or [download](https://github.com/cubed-dev/cubed/blob/main/examples/add-asarray.py) from GitHub, then run with:

```shell
python add-asarray.py
```

If successful it will print a 4x4 array:

```
[[ 2  4  6  8]
 [10 12 14 16]
 [18 20 22 24]
 [26 28 30 32]]
 ```

## Adding two larger arrays

The next example generates two random 20GB arrays and then adds them together.

```{eval-rst}
.. literalinclude:: ../../examples/add-random.py
```

Paste the code into a file called `add-random.py`, or [download](https://github.com/cubed-dev/cubed/blob/main/examples/add-random.py) from GitHub, then run with:

```shell
python add-random.py
```

This example demonstrates how we can use callbacks to gather information about the computation.

- `RichProgressBar` shows a progress bar for the computation as it is running.
- `TimelineVisualizationCallback` produces a plot (after the computation has completed) showing the timeline of events in the task lifecycle.
- `HistoryCallback` produces various stats about the computation once it has completed.

The plots and stats are written in the `history` directory in a directory with a timestamp. You can open the latest plot with

```shell
open $(ls -d history/compute-* | tail -1)/timeline.svg
```

## Matmul

The next example generates two random 5GB arrays and then multiplies them together. This is a more intensive computation than addition, and will take a few minutes to run locally.

```{eval-rst}
.. literalinclude:: ../../examples/matmul-random.py
```

Paste the code into a file called `matmul-random.py`, or [download](https://github.com/cubed-dev/cubed/blob/main/examples/matmul-random.py) from GitHub, then run with:

```shell
python matmul-random.py
```

## Trying different executors

You can run these scripts using different executors by setting environment variables to control the Cubed configuration.

For example, this will use the `processes` executor to run the example:

```shell
CUBED_SPEC__EXECUTOR_NAME=processes python add-random.py
```

For cloud executors, it's usually best to put all of the configuration in one YAML file, and set the `CUBED_CONFIG` environment variable to point to it:

```shell
export CUBED_CONFIG=/path/to/lithops/aws/cubed.yaml
python add-random.py
```

You can read more about how [configuration](../configuration.md) works in Cubed in general, and detailed steps to run on a particular cloud service [here](#cloud-set-up).
