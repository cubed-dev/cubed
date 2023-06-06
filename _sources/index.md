# Cubed

## Bounded-memory serverless distributed N-dimensional array processing

Cubed is a distributed N-dimensional array library implemented in Python using bounded-memory serverless processing and Zarr for storage.

- Implements the [Python Array API standard](https://data-apis.org/array-api/latest/)
- Guaranteed maximum memory usage for standard array functions
- Follows [Dask Array](https://docs.dask.org/en/stable/array.html)'s chunked array API (`map_blocks`, `rechunk`, etc)
- [Zarr](https://zarr.readthedocs.io/en/stable/) for storage
- Multiple serverless runtimes: Python (in-process), [Lithops](https://lithops-cloud.github.io/), [Modal](https://modal.com/), [Apache Beam](https://beam.apache.org/)

## Documentation

```{toctree}
---
maxdepth: 2
caption: For users
---
getting-started/index
user-guide/index
Examples <https://github.com/tomwhite/cubed/tree/main/examples/README.md>
api
array-api
related-projects
```

```{toctree}
---
maxdepth: 2
caption: For developers
---
design
operations
computation
contributing
```
