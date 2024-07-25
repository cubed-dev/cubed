# Rechunk a virtual dataset

This example demonstrates how to rechunk a collection of necdf files on s3
into a single zarr store.

First, lithops and Virtualizarr construct a virtual dataset comprised of the
netcdf files on s3. Then, xarray-cubed rechunks the virtual dataset into a
zarr.

## Credits
Inspired by Pythia's cookbook: https://projectpythia.org/kerchunk-cookbook
by norlandrhagen.

Please, contribute improvements.



1. Set up a Python environment
```bash
conda create --name virtualizarr-rechunk -y python=3.11
conda activate virtualizarr-rechunk
pip install -r requirements.txt
```

1. Set up cubed executor for [lithops-aws](https://github.com/cubed-dev/cubed/blob/main/examples/lithops/aws/README.md) by editing `./lithops.yaml` with your `bucket` and `execution_role`.
```bash

1. Build a runtime image for Cubed
```bash
export LITHOPS_CONFIG_FILE=$(pwd)/lithops.yaml
export CUBED_CONFIG=$(pwd)
lithops runtime build -b aws_lambda -f Dockerfile_virtualizarr virtualizarr-runtime
```

1. Run the script
```bash
python cubed-rechunk.py
```

## Cleaning up
To rebuild the Litops image, delete the existing one by running
```bash
lithops runtime delete -b aws_lambda -d virtualizarr-runtime
```
