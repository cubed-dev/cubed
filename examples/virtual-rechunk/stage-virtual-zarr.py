# Use lithops to construct a virtual zarr from netcdf files on s3.

import fsspec
import lithops
import os
import xarray as xr

from virtualizarr import open_virtual_dataset

bucket_url = os.getenv("BUCKET_URL")

fs_read = fsspec.filesystem("s3", anon=True, skip_instance_cache=True)
files_paths = fs_read.glob("s3://wrf-se-ak-ar5/ccsm/rcp85/daily/2060/*")
file_pattern = sorted(["s3://" + f for f in files_paths])

# Truncate file_pattern while debugging
file_pattern = file_pattern[:4]

print(f"{len(file_pattern)} file paths were retrieved.")


def map_references(fil):
    """ Map function to open virtual datasets.
    """
    vds = open_virtual_dataset(
        fil,
        indexes={},
        loadable_variables=['Time'],
        cftime_variables=['Time'],
    )
    return vds


def reduce_references(results):
    """ Reduce to concat virtual datasets.
    """
    combined_vds = xr.combine_nested(
        results,
        concat_dim=["Time"],
        coords="minimal",
        compat="override",
    )
    
    return combined_vds


fexec = lithops.FunctionExecutor(config_file="lithops.yaml")

futures = fexec.map_reduce(
    map_references,
    file_pattern,
    reduce_references,
    spawn_reducer=100,
)

ds = futures.get_result()

# Save the virtual zarr manifest
ds.virtualize.to_kerchunk(f"combined.json", format="json")

# Upload manifest to s3
fs_write = fsspec.filesystem("s3", anon=False, skip_instance_cache=True)
fs_write.put("combined.json", f"{bucket_url}/combined.json")
