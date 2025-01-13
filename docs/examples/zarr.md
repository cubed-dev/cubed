---
file_format: mystnb
kernelspec:
  name: python3
---
# Zarr

Cubed was designed to work seamlessly with Zarr data. The examples below demonstrate using {py:func}`cubed.from_zarr`, {py:func}`cubed.to_zarr` and {py:func}`cubed.store` to read and write Zarr data.

## Write to Zarr

We'll start by creating a small chunked array containing random data in Cubed and writing it to Zarr using {py:func}`cubed.to_zarr`. Note that the call to `to_zarr` executes eagerly.

```{code-cell} ipython3
import cubed
import cubed.random

# 2MB chunks
a = cubed.random.random((5000, 5000), chunks=(500, 500))

# write to Zarr
cubed.to_zarr(a, "a.zarr")
```

## Read from Zarr

We can check that the Zarr file was created by loading it from disk using {py:func}`cubed.from_zarr`:

```{code-cell} ipython3
cubed.from_zarr("a.zarr")
```

## Multiple arrays

To write multiple arrays in a single computation use {py:func}`cubed.store`:

```{code-cell} ipython3
import cubed
import cubed.random

# 2MB chunks
a = cubed.random.random((5000, 5000), chunks=(500, 500))
b = cubed.random.random((5000, 5000), chunks=(500, 500))

# write to Zarr
arrays = [a, b]
paths = ["a.zarr", "b.zarr"]
cubed.store(arrays, paths)
```

Then to read the Zarr files back, we use {py:func}`cubed.from_zarr` for each array and perform whatever array operations we like on them. Only when we call `to_zarr` is the whole computation executed.

```{code-cell} ipython3
import cubed.array_api as xp

# read from Zarr
a = cubed.from_zarr("a.zarr")
b = cubed.from_zarr("b.zarr")

# perform operation
c = xp.add(a, b)

# write to Zarr
cubed.to_zarr(c, store="c.zarr")
```
