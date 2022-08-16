# Cubed

## Bounded-memory serverless distributed N-dimensional array processing

Cubed is a distributed N-dimensional array library implemented in Python using bounded-memory serverless processing and Zarr for storage.

- Implements the [Python Array API standard](https://data-apis.org/array-api/latest/)
- Guaranteed maximum memory usage for standard array functions
- [Zarr](https://zarr.readthedocs.io/en/stable/) for storage
- Multiple serverless runtimes: Python (in-process), [Lithops](https://lithops-cloud.github.io/), [Modal](https://modal.com/), [Apache Beam](https://beam.apache.org/)

## Documentation

```{toctree}
---
maxdepth: 3
---
api
array_api
```
