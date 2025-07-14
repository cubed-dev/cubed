---
file_format: mystnb
kernelspec:
  name: python3
---
# Icechunk

This example shows how to perform large-scale distributed writes to Icechunk using Cubed
(based on the examples for using [Icechunk with Dask](https://icechunk.io/en/latest/dask/)).

Install the package pre-requisites by running the following:

```shell
pip install cubed icechunk
```

Start by creating an Icechunk store.

```{code-cell} ipython3
import icechunk
import tempfile

# initialize the icechunk store
storage = icechunk.local_filesystem_storage(tempfile.TemporaryDirectory().name)
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")
```

## Write to Icechunk

Use `cubed.icechunk.store_icechunk` to write a Cubed array to an Icechunk store.
The API follows that of {py:func}`cubed.store`.

First create a Cubed array to write:

```{code-cell} ipython3
import cubed
shape = (100, 100)
cubed_chunks = (20, 20)
cubed_array = cubed.random.random(shape, chunks=cubed_chunks)
```

Now create the Zarr array you will write to.

```{code-cell} ipython3
import zarr

zarr_chunks = (10, 10)
group = zarr.group(store=session.store, overwrite=True)

zarray = group.create_array(
    "array",
    shape=shape,
    chunks=zarr_chunks,
    dtype="f8",
    fill_value=float("nan"),
)
session.commit("initialize array")
```

Note that the chunks in the store are a divisor of the Cubed chunks. This means each individual write task is independent, and will not conflict. It is your responsibility to ensure that such conflicts are avoided.

First remember to fork the session before re-opening the Zarr array. `store_icechunk` will merge all the remote write sessions on the cluster before returning back a single merged `ForkSession`.

```{code-cell} ipython3
from cubed.icechunk import store_icechunk

session = repo.writable_session("main")
fork = session.fork()
zarray = zarr.open_array(fork.store, path="array")
remote_session = store_icechunk(
    sources=[cubed_array],
    targets=[zarray]
)
```

Merge the remote session in to the local Session
```{code-cell} ipython3
session.merge(remote_session)
```


Finally commit your changes!

```{code-cell} ipython3
print(session.commit("wrote a cubed array!"))
```

## Read from Icechunk

Use {py:func}`cubed.from_zarr` to read from Icechunk - note that no special Icechunk-specific function is needed in this case.

```{code-cell} ipython3
cubed.from_zarr(store=session.store, path="array")
```
