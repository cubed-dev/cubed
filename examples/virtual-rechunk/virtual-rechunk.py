# Rechunk a collection of necdf files on s3 into a single zarr store.
#
# First, lithops and Virtualizarr construct a virtual dataset comprised of the
# netcdf files on s3. Then, xarray-cubed rechunks the virtual dataset into a
# zarr.
#
# Inspired by Pythia's cookbook: https://projectpythia.org/kerchunk-cookbook
# by norlandrhagen.
#
# Please, contribute improvements.

import fsspec
import lithops
import xarray as xr

from virtualizarr import open_virtual_dataset

fs_read = fsspec.filesystem("s3", anon=True, skip_instance_cache=True)
files_paths = fs_read.glob("s3://wrf-se-ak-ar5/ccsm/rcp85/daily/2060/*")
file_pattern = sorted(["s3://" + f for f in files_paths])

# truncate file_pattern while debugging
file_pattern = file_pattern[:4]

print(f"{len(file_pattern)} file paths were retrieved.")


def map_references(fil):
    """ Map function to open virtual datasets.
    """
    vds = open_virtual_dataset(fil,
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
        concat_dim=['Time'],
        coords='minimal',
        compat='override',
    )
    # possibly write parquet to s3 here
    return combined_vds


fexec = lithops.FunctionExecutor(config_file="lithops.yaml")

futures = fexec.map_reduce(
    map_references,
    file_pattern,
    reduce_references,
    spawn_reducer=100,
)

ds = futures.get_result()
ds.virtualize.to_kerchunk('combined.json', format='json')

# NOTE: In jupyter, open_dataset seems to cache the json, such that changes
# aren't propogated until the kernel is restarted.
combined_ds = xr.open_dataset('combined.json',
                              engine="kerchunk",
                              chunks={},
                              chunked_array_type='cubed',
                              )

combined_ds['Time'].attrs = {}  # to_zarr complains about attrs

rechunked_ds = combined_ds.chunk(
    chunks={'Time': 5, 'south_north': 25, 'west_east': 32}
)

rechunked_ds.to_zarr('rechunked.zarr',
                     mode='w',
                     encoding={},  # TODO
                     consolidated=True,
                     safe_chunks=False,
                     )
