# Computation

## Plan

Cubed has a lazy computation model. As array functions are invoked, a computation _plan_ is built up, and it is only executed when explicitly triggered with
a call to `compute`, or when implicitly triggered by converting an array to an in-memory (NumPy) or on-disk (Zarr) representation.

A `Plan` object is a directed acyclic graph (DAG), where the nodes are arrays and the edges express primitive operations. For example, one array may be rechunked to another using a `rechunk` operation. Or a pair of arrays may be added together using a `blockwise` operation.

## Memory

The primitive operations, `blockwise` and `rechunk` both have memory requirements that are known ahead of time. Each operation runs a _task_ to compute each chunk of the output. The memory needed for each task is a function of chunk size, dtype, and the nature of the operation, which can be computed while building the plan.

See <project:user-guide/memory.md> for discussion of memory settings.

## Execution

A plan is executed by traversing the DAG and materializing arrays by writing them to Zarr storage. Details of how a plan is executed depends on the runtime. Distributed runtimes, for example, may choose to materialize arrays that don't depend on one another in parallel for efficiency.

This processing model has advantages and disadvantages. One advantage is that since there is no shuffle involved, it is a straightforward model that can scale up with very high-levels of parallelism - for example in a serverless environment. This also makes it straightforward to make it run on multiple execution engines.

The main disadvantage of this model is that every intermediate array is written to storage, which can be slow. However, there are opportunities to optimize the DAG before running it (such as map fusion).
