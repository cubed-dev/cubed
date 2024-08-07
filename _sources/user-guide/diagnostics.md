# Diagnostics

Cubed provides a variety of tools to understand a computation before running it, to monitor its progress while running, and to view performance statistics after it has completed.

To use these features ensure that the optional dependencies for diagnostics have been installed:

```shell
python -m pip install "cubed[diagnostics]"
```

## Visualize the computation plan

Before running a computation, Cubed will create an internal plan that it uses to compute the output arrays.

The plan is a directed acyclic graph (DAG), and it can be useful to visualize it to see the number of steps involved in your computation, the number of tasks in each step (and overall), and the amount of intermediate data written out.

The {py:meth}`Array.visualize() <cubed.Array.visualize()>` method on an array creates an image of the DAG. By default it is saved in a file called *cubed.svg* in the current working directory, but the filename and format can be changed if needed. If running in a Jupyter notebook the image will be rendered in the notebook.

If you are computing multiple arrays at once, then there is a {py:func}`visualize <cubed.visualize>` function that takes multiple array arguments.

This example shows a tiny computation and the resulting plan:

```python
import cubed.array_api as xp
import cubed.random

a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
b = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
c = xp.add(a, b)

c.visualize()
```

![Cubed visualization of a tiny computation](../images/cubed-add.svg)

There are two type of nodes in the plan. Boxes with rounded corners are operations, while boxes with square corners are arrays.

In this case there are three operations (labelled `op-001`, `op-002`, and `op-003`), which produce the three arrays `a`, `b`, and `c`. (There is always an additional operation called `create-arrays`, shown on the right, which Cubed creates automatically.)

Array `c` is coloured orange, which means it is materialized as a Zarr array. Arrays `a` and `b` do not need to be materialized as Zarr arrays since they are small constant arrays that are passed to the workers running the tasks.

Similarly, the operation that produces `c` is shown in a lilac colour to signify that it runs tasks to produce the output. Operations `op-001` and `op-002` don't run any tasks since `a` and `b` are just small constant arrays.

## Progress bar

You can display a progress bar to track your computation by passing callbacks to {py:meth}`compute() <cubed.Array.compute()>`:

```ipython
>>> from cubed.diagnostics.rich import RichProgressBar
>>> progress = RichProgressBar()
>>> c.compute(callbacks=[progress])  # c is the array from above
  create-arrays 1/1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:00
  op-003 add    4/4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:00
```

This will work in Jupyter notebooks, and for all executors.

You can also pass callbacks to functions that call `compute`, such as {py:func}`store <cubed.store>` or {py:func}`to_zarr <cubed.to_zarr>`.

## History and timeline visualization

The history and timeline visualization callbacks can be used to find out how long tasks took to run, and how much memory they used.

The timeline visualization is useful to determine how much time was spent in worker startup, as well as how much stragglers affected the overall time of the computation. (Ideally, we want vertical lines on this plot, which would represent perfect horizontal scaling.)

See the [examples](https://github.com/cubed-dev/cubed/blob/main/examples/README.md) for more information about how to use them.
