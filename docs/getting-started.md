# Getting Started

We'll start with a simple example that runs locally.

## Installation

Install Cubed with pip:

```shell
pip install cubed
```

This installs a minimal set of dependencies for running Cubed, which is sufficient for the demo below. You can also install the `diagnostics` extra package, which is needed for later examples to provide things like progress bars and visualizations of the computation:

```shell
pip install "cubed[diagnostics]"
```

Alternatively, you can install Cubed with Conda (note that this doesn't include the packages for diagnostics):

```shell
conda install -c conda-forge cubed
```

## Demo

First, we'll create a small array `a`:

```python
import cubed
import cubed.array_api as xp
spec = cubed.Spec(work_dir="tmp", allowed_mem="100kB")
a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
```

Cubed implements the [Python Array API standard](https://data-apis.org/array-api/latest/), which is essentially a subset of NumPy, and is imported as `xp` by convention.

Notice that we also specify chunks, just like in Dask Array, and a {py:class}`Spec <cubed.Spec>` object that describes the resources available to run computations.

Next we create another array `b` and add to two array together to get `c`.

```python
b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2), spec=spec)
c = xp.add(a, b)
```

Cubed uses lazy evaluation, so nothing has been computed yet.

```python
c.compute()
```

This runs the computation using the (default) local Python executor and prints the result (if run interactively):

```
array([[ 2,  3,  4],
       [ 5,  6,  7],
       [ 8,  9, 10]])
```

That's it! For your next step you can read the [user guide](user-guide/index.md), have a look at [configuration](configuration.md) options, or see more [examples](https://github.com/cubed-dev/cubed/blob/main/examples/README.md) to run locally or in the cloud.
