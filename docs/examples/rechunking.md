---
file_format: mystnb
kernelspec:
  name: python3
---
# Rechunking

This example uses Xarray to rechunk a dataset.

Install the package pre-requisites by running the following:

```shell
pip install cubed cubed-xarray xarray pooch netCDF4
```

## Open dataset

Start by importing Xarray - note that we don't need to import Cubed or `cubed-xarray`, since they will be picked up automatically.

```{code-cell} ipython3
import xarray as xr

xr.set_options(display_expand_attrs=False);
```

We open an Xarray dataset (in netCDF format) using the usual `open_dataset` function. By specifying `chunks={}` we ensure that the dataset is chunked using the on-disk chunking (here it is the netCDF file chunking). The `chunked_array_type` argument specifies which chunked array type to use - Cubed in this case.

```{code-cell} ipython3
ds = xr.tutorial.open_dataset(
    "air_temperature", chunked_array_type="cubed", chunks={}
)
```

The `air` data variable is a `cubed.Array`, and we can see that this small dataset has a single on-disk chunk.

```{code-cell} ipython3
ds["air"]
```

## Rechunk

To change the chunking we use Xarray's `chunk` function:

```{code-cell} ipython3
rds = ds.chunk({'time':1}, chunked_array_type="cubed")
```

Looking at the `air` data variable again, we can see that it is now chunked along the time dimension.

```{code-cell} ipython3
rds["air"]
```

## Save to Zarr

Since Cubed has a lazy computation model, the data has not been loaded from disk yet. We can save a copy of the rechunked dataset by calling `to_zarr`:

```{code-cell} ipython3
rds.to_zarr("rechunked_air_temperature.zarr", mode="w", consolidated=True);
```

This will run a computation that loads the input data and writes it out to a Zarr store on the local filesystem with the new chunking. We can check that it worked by re-loading from disk using `xarray.open_dataset` and checking that the chunks are the same:

```{code-cell} ipython3
ds = xr.open_dataset(
    "rechunked_air_temperature.zarr", chunked_array_type="cubed", chunks={}
)
assert ds.chunks == rds.chunks
```
