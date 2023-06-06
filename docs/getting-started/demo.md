# Demo

We'll start with a simple example that runs locally.

```python
import cubed
import cubed.array_api as xp
spec = cubed.Spec(work_dir="tmp", allowed_mem="100kB")
a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
```

Cubed implements the [Python Array API standard](https://data-apis.org/array-api/latest/), which is essentially a subset of NumPy, and is imported as `xp` by convention.

Notice that we also specify chunks, just like in Dask Array, and a {py:class}`Spec <cubed.Spec>` object that describes the resources available to run the computation.

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

See the [examples README](https://github.com/tomwhite/cubed/tree/main/examples/README.md) for examples that run on cloud services.
