---
file_format: mystnb
kernelspec:
  name: python3
---
# Xarray

Cubed can work with Xarray datasets via the [`cubed-xarray`](https://github.com/cubed-dev/cubed-xarray) package.

Install by running the following:

```shell
pip install cubed cubed-xarray xarray pooch netCDF4
```

Note that `pooch` and `netCDF4` are needed to access the Xarray tutorial datasets that we use in the example below.

## Open dataset

Start by importing Xarray - note that we don't need to import Cubed or `cubed-xarray`, since they will be picked up automatically.

```{code-cell} ipython3
import xarray as xr

xr.set_options(display_expand_attrs=False, display_expand_data=True);
```

We open an Xarray dataset (in netCDF format) using the usual `open_dataset` function. By specifying `chunks={}` we ensure that the dataset is chunked using the on-disk chunking (here it is the netCDF file chunking). The `chunked_array_type` argument specifies which chunked array type to use - Cubed in this case.

```{code-cell} ipython3
ds = xr.tutorial.open_dataset(
    "air_temperature", chunked_array_type="cubed", chunks={}
)
ds
```

Notice that the `air` data variable is a `cubed.Array`. Since Cubed has a lazy computation model, this array is not loaded from disk until a computation is run.

## Convert to Zarr

We can use Cubed to convert the dataset to Zarr format by calling `to_zarr` on the dataset:

```{code-cell} ipython3
ds.to_zarr("air_temperature_cubed.zarr", mode="w", consolidated=True);
```

This will run a computation that loads the input data and writes it out to a Zarr store on the local filesystem.

## Compute the mean

We can also use Xarray's API to run computations on the dataset using Cubed. Here we find the mean air temperature over time, for each location:

```{code-cell} ipython3
mean = ds.air.mean("time", skipna=False)
mean
```

To run the computation we need to call `compute`:

```{code-cell} ipython3
mean.compute()
```

This is fine for outputs that fit in memory like the example here, but sometimes we want to write the output of the computation to Zarr, which we do by calling `to_zarr` on the dataset instead of `compute`:

```{code-cell} ipython3
mean.to_zarr("mean_air_temperature.zarr", mode="w", consolidated=True);
```

We can check that the Zarr file was created by loading it from disk using `xarray.open_dataset`:

```{code-cell} ipython3
xr.open_dataset(
    "mean_air_temperature.zarr", chunked_array_type="cubed", chunks={}
)
```
