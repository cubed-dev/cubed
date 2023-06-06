# Why Cubed?

Managing memory is one of the major challenges in designing and running distributed systems. Computation frameworks like Apache Hadoop's MapReduce, Apache Spark, Beam, and Dask all provide a general-purpose processing model, which has lead to their widespread adoption. Successful use at scale however requires the user to carefully configure memory for worker nodes, and to understand how work is allocated to workers, which breaks the high-level programming abstraction. A disproportionate amount of time is often spent tuning the memory configuration of a large computation.

A common theme here is that most interesting computations are not embarrassingly parallel, but involve shuffling data between nodes. A lot of engineering effort has been put into optimizing the shuffle in Hadoop, Spark, Beam (Google Dataflow), and to a lesser extent Dask. This has undoubtedly improved performance, but has not made the memory problems go away.

Another approach has started gaining traction in the last few years. [Lithops](https://lithops-cloud.github.io/) (formerly Pywren) and [Rechunker](https://rechunker.readthedocs.io/), eschew centralized systems like the shuffle, and do everything via serverless cloud services and cloud storage.

Rechunker is interesting, since it implements a very targeted use case (rechunking persistent N-dimensional arrays), using only stateless (serverless) operations, with guaranteed memory usage. Even though it can run on systems like Beam and Dask, it deliberately avoids passing array chunks between worker nodes using the shuffle. Instead, all bulk data operations are reads from, or writes to, cloud storage (Zarr in this case). Since chunks are always of known size it is possible to tightly control memory usage, thereby avoiding unpredictable memory use at runtime.

Cubed is an attempt to go further: implement all distributed array operations using a bounded-memory serverless model.
