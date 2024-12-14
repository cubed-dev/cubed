# pip install virtualizarr pooch netcdf4 h5py h5netcdf icechunk 'zarr==3.0.0b2'
# pip install -U git+https://github.com/mpiannucci/kerchunk@v3
# pip install 'cubed[diagnostics]' cubed-xarray
# pip install 'jupyter[notebook]'

import xarray as xr

# create an example pre-existing netCDF4 file
ds = xr.tutorial.open_dataset("air_temperature")
print(ds)
ds.to_netcdf("air.nc")

from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset("air.nc")
print(vds)

marr = vds["air"].data
print(marr)
manifest = marr.manifest
print(manifest.dict())
