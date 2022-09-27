# Cubed

## Bounded-memory serverless distributed N-dimensional array processing

Cubed is a distributed N-dimensional array library implemented in Python using bounded-memory serverless processing and Zarr for storage.

- Implements the [Python Array API standard](https://data-apis.org/array-api/latest/)
- Guaranteed maximum memory usage for standard array functions
- Follows [Dask Array](https://docs.dask.org/en/stable/array.html)'s chunked array API (`map_blocks`, `rechunk`, etc)
- [Zarr](https://zarr.readthedocs.io/en/stable/) for storage
- Multiple serverless runtimes: Python (in-process), [Lithops](https://lithops-cloud.github.io/), [Modal](https://modal.com/), [Apache Beam](https://beam.apache.org/)

## Motivation

Managing memory is one of the major challenges in designing and running distributed systems. Computation frameworks like Apache Hadoop's MapReduce, Apache Spark, Beam, and Dask all provide a general-purpose processing model, which has lead to their widespread adoption. Successful use at scale however requires the user to carefully configure memory for worker nodes, and to understand how work is allocated to workers, which breaks the high-level programming abstraction. A disproportionate amount of time is often spent tuning the memory configuration of a large computation.

A common theme here is that most interesting computations are not embarrassingly parallel, but involve shuffling data between nodes. A lot of engineering effort has been put into optimizing the shuffle in Hadoop, Spark, Beam (Google Dataflow), and to a lesser extent Dask. This has undoubtedly improved performance, but has not made the memory problems go away.

Another approach has started gaining traction in the last few years. [Lithops](https://lithops-cloud.github.io/) (formerly Pywren) and [Rechunker](https://rechunker.readthedocs.io/), eschew centralized systems like the shuffle, and do everything via serverless cloud services and cloud storage.

Rechunker is interesting, since it implements a very targeted use case (rechunking persistent N-dimensional arrays), using only stateless (serverless) operations, with guaranteed memory usage. Even though it can run on systems like Beam and Dask, it deliberately avoids passing array chunks between worker nodes using the shuffle. Instead, all bulk data operations are reads from, or writes to, cloud storage (Zarr in this case). Since chunks are always of known size it is possible to tightly control memory usage, thereby avoiding unpredictable memory use at runtime.

This project is an attempt to go further: implement all distributed array operations using a bounded-memory serverless model.

## Documentation

```{toctree}
---
maxdepth: 2
caption: For users
---
getting_started
api
array_api
related_projects
Examples <https://github.com/tomwhite/cubed/tree/main/examples/README.md>
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
