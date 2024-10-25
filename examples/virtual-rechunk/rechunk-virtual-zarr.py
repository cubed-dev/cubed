# Rechunk a virtual zarr on s3 into a single zarr store using xarray-cubed.
#
# Prior to running this script, create the virtual zarr with
# > python create-virtualzarr.py
#
# NOTE: In jupyter, open_dataset seems to cache the json, such that changes
# aren't propogated until the kernel is restarted.

import os
import xarray as xr


bucket_url = os.getenv("BUCKET_URL")

combined_ds = xr.open_dataset(
    f"{bucket_url}/combined.json", # location must be accessible to workers
    engine="kerchunk",
    chunks={},
    chunked_array_type="cubed",
)

combined_ds['Time'].attrs = {}  # otherwise to_zarr complains about attrs

rechunked_ds = combined_ds.chunk(
    chunks={'Time': 5, 'south_north': 25, 'west_east': 32},
    chunked_array_type="cubed",
)

rechunked_ds.to_zarr(
    f"{bucket_url}/rechunked.zarr",
    mode="w",
    encoding={},  # TODO
    consolidated=True,
    safe_chunks=False,
)
