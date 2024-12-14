# pip install virtualizarr pooch netcdf4 h5py h5netcdf kerchunk zarr
# pip install 'cubed[diagnostics]' cubed-xarray
# pip install 'jupyter[notebook]'

# For icechunk (not working)
# pip install 'zarr==3.0.0b2' icechunk
# pip install -U git+https://github.com/mpiannucci/kerchunk@v3


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
