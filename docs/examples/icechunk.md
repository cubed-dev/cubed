---
file_format: mystnb
kernelspec:
  name: python3
---
# Icechunk

This example shows how to perform large-scale distributed writes to Icechunk using Cubed
(based on the examples for using [Icechunk with Dask](https://icechunk.io/en/latest/icechunk-python/dask/)).

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
icechunk_repo = icechunk.Repository.create(storage)
icechunk_session = icechunk_repo.writable_session("main")
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
group = zarr.group(store=icechunk_session.store, overwrite=True)

zarray = group.create_array(
    "array",
    shape=shape,
    chunks=zarr_chunks,
    dtype="f8",
    fill_value=float("nan"),
)
```

Note that the chunks in the store are a divisor of the Cubed chunks. This means each individual write task is independent, and will not conflict. It is your responsibility to ensure that such conflicts are avoided.

Now write

```{code-cell} ipython3
from cubed.icechunk import store_icechunk

store_icechunk(
    icechunk_session,
    sources=[cubed_array],
    targets=[zarray]
)
```

Finally commit your changes!

```{code-cell} ipython3
print(icechunk_session.commit("wrote a cubed array!"))
```

## Read from Icechunk

Use {py:func}`cubed.from_zarr` to read from Icechunk - note that no special Icechunk-specific function is needed in this case.

```{code-cell} ipython3
cubed.from_zarr(store=icechunk_session.store, path="array")
```

## Distributed writes

In distributed contexts where the Session, and Zarr Array objects are sent across the network, you must opt-in to successful pickling of a writable store.
`cubed.icechunk.store_icechunk` takes care of the hard bit of merging Sessions but it is required that you opt-in to pickling prior to creating the target Zarr array objects.

Here is an example:

```{code-cell} ipython3
from cubed import config

# start a new session. Old session is readonly after committing

icechunk_session = icechunk_repo.writable_session("main")
zarr_chunks = (10, 10)

# use the Cubed processes executor which requires pickling
with config.set({"spec.executor_name": "processes"}):
    with icechunk_session.allow_pickling():
        cubed_array = cubed.random.random(shape, chunks=cubed_chunks)

        group = zarr.group(
            store=icechunk_session.store,
            overwrite=True
        )

        zarray = group.create_array(
            "array",
            shape=shape,
            chunks=zarr_chunks,
            dtype="f8",
            fill_value=float("nan"),
        )

        store_icechunk(
            icechunk_session,
            sources=[cubed_array],
            targets=[zarray]
        )

print(icechunk_session.commit("wrote a cubed array!"))
```
