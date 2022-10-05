# Computation

## Plan

Cubed has a lazy computation model. As array functions are invoked, a computation _plan_ is built up, and it is only executed when explicitly triggered with
a call to `compute`, or when implicitly triggered by converting an array to an in-memory (NumPy) or on-disk (Zarr) representation.

A `Plan` object is a directed acyclic graph (DAG), where the nodes are arrays and the edges express primitive operations. For example, one array may be rechunked to another using a `rechunk` operation. Or a pair of arrays may be added together using a `blockwise` operation.

## Memory

The primitive operations, `blockwise` and `rechunk` both have memory requirements that are known ahead of time. Each operation runs a _task_ to compute each chunk of the output. The memory needed for each task is a function of chunk size, dtype, and the nature of the operation, which can be computed while building the plan.

While it is not possible in general to compute the precise amount of memory that will be used (since it is not known ahead of time how well a Zarr chunk will compress), it is possible to put a conservative upper bound on the memory usage. This upper bound is the `required_mem` for a task, and it is calculated automatically by Cubed.

![Memory](images/memory.svg)

The maximum amount of memory that tasks are allowed to use must be specified by the user. This is done by setting the `reserved_mem` and `max_mem` parameters on the `Spec` object. The `reserved_mem` is the amount of memory reserved on a worker for non-data use - it's whatever is needed by the Python process for running a task, and can be estimated using the `measure_reserved_memory` function. The `max_mem` setting is the memory available to a worker for data use - for loading arrays into memory and processing them. It can be set to be the total memory available to a worker, less the `reserved_mem`.

If the `required_mem` calculated by Cubed is greater than the value of `max_mem` set by the user, an exception is raised during the planning phase. This check means that the user can have high confidence that the operation will run within its memory budget.

## Execution

A plan is executed by traversing the DAG and materializing arrays by writing them to Zarr storage. Details of how a plan is executed depends on the runtime. Distributed runtimes, for example, may choose to materialize arrays that don't depend on one another in parallel for efficiency.

This processing model has advantages and disadvantages. One advantage is that since there is no shuffle involved, it is a straightforward model that can scale up with very high-levels of parallelism - for example in a serverless environment. This also makes it straightforward to make it run on multiple execution engines.

The main disadvantage of this model is that every intermediate array is written to storage, which can be slow. However, there are opportunities to optimize the DAG before running it (such as map fusion).

## Runtime Features

<dl>
  <dt>Task failure handling</dt>
  <dd>If a task fails - with an IO exception when reading or writing to Zarr, for example - it will be retried (up to a total of three attempts).</dd>
  <dt>Resume a computation from a checkpoint</dt>
  <dd>Since intermediate arrays are persisted to Zarr, it is possible to resume a computation without starting from scratch. To do this, the Cubed <code>Array</code> object should be stored persistently (using <code>dill</code>), so it can be reloaded in a new process and then <code>compute()</code> called on it to finish the computation.</dd>
  <dt>Straggler mitigation</dt>
  <dd>A few slow running tasks (called stragglers) can disproportionately slow down the whole computation. To mitigate this, speculative duplicate tasks are launched in certain circumstances, acting as backups that complete more quickly than the straggler, hence bringing down the overall time taken.</dd>
</dl>
