# Cubed

**_Note: this is a proof-of-concept, and many things are incomplete or don't work._**

## Bounded-memory serverless distributed N-dimensional array processing

Cubed is a distributed N-dimensional array library implemented in Python using bounded-memory serverless processing and Zarr for storage.

- Implements the [Python Array API standard](https://data-apis.org/array-api/latest/) (see [coverage status](./api_status.md))
- Guaranteed maximum memory usage for standard array functions
- Follows [Dask Array](https://docs.dask.org/en/stable/array.html)'s chunked array API (`map_blocks`, `rechunk`, `apply_gufunc`, etc)
- [Zarr](https://zarr.readthedocs.io/en/stable/) for persistent and intermediate storage
- Multiple serverless runtimes: Python (in-process), [Lithops](https://lithops-cloud.github.io/), [Modal](https://modal.com/), [Apache Beam](https://beam.apache.org/)
- Integration with [Xarray](https://xarray.dev/) via [cubed-xarray](https://github.com/xarray-contrib/cubed-xarray)

[Documentation](https://tomwhite.github.io/cubed)
