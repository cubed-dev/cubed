# Scaling

Cubed is specifically designed to maintain optimal performance while processing large arrays (of order Terabytes so far).
This page aims to provide a deeper understanding of how Cubed's design scales in both theoretical and practical scenarios.

## Preface: Types of Scaling

There are different types of scaling to consider in distributed computing:

**Horizontal versus Vertical Scaling**: Horizontal scaling refers to adding more machines to the system to improve its throughput, while vertical scaling means upgrading an existing machine to a larger one with more speed and resources.

**Weak versus Strong Scaling**: Strong scaling is defined as how the solution time varies with the number of processors for a fixed total problem size.
Weak scaling is defined as how the solution time varies with the number of processors for a fixed problem size per processor.
In other words, strong scaling measures how much faster you can get a given problem done, whereas weak scaling measures how big a problem you can get done in a reasonable time.

Cubed has been designed to scale horizontally, in such a way as to display good weak scaling even when processing large array datasets.


## Theoretical vs Practical Scaling of Cubed

To understand factors affecting the scaling and performance of Cubed, we will start by considering the simplest possible calculation and add complexity.

### Single-step Calculation

The  simplest non-trivial operation in Cubed would map one block from the input array to one block in the output array, with no intermediate persistent stores.
Changing the sign of every element in an array would be an example of this type of operation, known as an [elementwise](#elemwise-operation) operation.

In an ideal environment where the serverless service provides infinite workers, the limiting factor for scaling would be concurrent writes to Zarr.
In such a case weak scaling should be linear, i.e. an array with more blocks could be processed in the same amount of time given proportionally more workers to process those blocks.

In practice, this ideal scenario may not be achieved, for a number of reasons.
Firstly, you need to make sure you're using a parallel executor, not the default single-threaded executor.
You also need enough parallelism to match the scale of your problem.
Weak scaling requires more workers than output chunks, so for large problems it might be necessary to adjust the executor's configuration to not restrict the ``max_workers``.
With fewer workers than chunks we would expect linear strong scaling, as every new worker added has nothing to wait for.

Stragglers are tasks that take much longer than average, who disproportionately hold up the next step of the computation.
Stragglers are handled by running backup tasks for any tasks that are running very slowly. This feature is enabled by default, but
if you need to turn it off you can do so with ``use_backups=False``.
Worker start-up time is another practical speed consideration, though it would delay computations of all scales equally.

### Multi-step Calculation

A multi-step calculation requires writing one or more intermediate arrays to persistent storage.
One important example in Cubed is the [rechunk](#rechunk-operation) operation, which guarantees bounded memory usage by writing and reading from one intermediate persistent Zarr store.

In multi-step calculations, the number of steps in the plan sets the minimum total execution time.
Hence, reducing the number of steps in the plan can lead to significant performance improvements.
Reductions can be carried out in fewer iterative steps if ``allowed_mem`` is larger.
Cubed automatically fuses some steps to enhance performance, but others (especially rechunk) cannot be fused without requiring a shuffle, which can potentially violate memory constraints.

In practical scenarios, stragglers can hold up the completion of each step separately, thereby cumulatively affecting the overall performance of the calculation.

### Multi-pipeline Calculation

A "pipeline" refers to an independent branch of the calculation.
For example, if you have two separate arrays to compute simultaneously, full parallelism requires sufficient workers for both tasks.
The same logic applies if you have two arrays feeding into a single array or vice versa.

```{note}
Currently Cubed will only execute independent pipelines in parallel if `compute_arrays_in_parallel=True` is passed to the executor function.
```


## Other Performance Considerations

### Different Executors

Different executors come with their own set of advantages and disadvantages.
For instance, some serverless executors have much faster worker startup times (such as Modal), while others may have different limits to the maximum workers.
Cluster-based executors (e.g. the Dask executor) will have different performance characteristics depending on how resources are provisioned for the cluster.
Some other executors such as Beam will first convert the Cubed Plan to a different representation before executing, which will affect performance.

### Different Cloud Providers

Different cloud providers' serverless offerings may perform differently. For example, Google Cloud Functions (GCF) has been observed to have more stragglers than AWS Lambda.


## Diagnosing Performance

To understand how your computation could perform better you first need to diagnose the source of any problems.

### Optimized Plan

Use {py:meth}`Plan.visualize() <cubed.Plan.visualize()>` to view the optimized plan. This allows you to see the number of steps involved in your calculation, the number of tasks in each step, and overall.

### History Callback

The history callback function can help determine how much time was spent in worker startup, as well as how much stragglers affected the overall speed.

### Timeline Visualization Callback

A timeline visualization callback can provide a visual representation of the above points. Ideally, we want vertical lines on this plot, which would represent perfect horizontal scaling.


## Tips

In Cubed, there are very few "magic numbers", meaning calculations generally take as long as they need to, with few other parameters to tune. Here are a few suggestions:

* Use {py:func}`measure_reserved_mem <cubed.measure_reserved_mem>`.
* Stick to ~100MB chunks.
* Set ``allowed_mem`` to around ~2GB (or larger if necessary).
* Use Cubed only for the part of the calculation where it's needed.
