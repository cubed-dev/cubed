# Related Projects

## Dask

Dask is a ["flexible library for parallel computing in Python"](https://docs.dask.org/en/latest/). It has several components: Dask Array, Dask DataFrame, Dask Bag, and Dask Delayed. Cubed supports only distributed arrays, corresponding to Dask Array.

Dask converts high-level operations into a task graph, where tasks correspond to chunk-level operations. In Cubed, the computational graph (or plan) is defined at the level of array operations and is decomposed into fine-grained tasks by the primitive operators for the runtime. Higher-level graphs are more useful for users, since they are easier to visualize and reason about. (Dask's newer High Level Graph abstraction is similar.)

Dask only has a single distributed runtime, Dask Distributed, whereas Cubed has the advantage of running on a variety of distributed runtimes, including more mature ones like Google Cloud Dataflow (a Beam runner).

The core operations and array API layers in Cubed are heavily influenced by Dask Array - in both naming, as well as implementation (Cubed uses Dask Array for some chunking utilities).

## Xarray

You can use Cubed with Xarray using the [cubed-xarray](https://github.com/xarray-contrib/cubed-xarray) package.

Read more about the integration in [Cubed: Bounded-memory serverless array processing in xarray](https://xarray.dev/blog/cubed-xarray).

## Previous work

This project is a continuation of a similar project I worked on, called [Zappy](https://github.com/lasersonlab/zappy/tree/master/zappy). What's changed in the intervening three years? Rechunker was created (I wasn't so concerned with memory when working on Zappy). The Python Array API standard has been created, which makes implementing a new array API less daunting than implementing the NumPy API. And I have a better understanding of Dask, and the fundamental nature of the blockwise operation.
