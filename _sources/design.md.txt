# Design

Cubed is composed of five layers: from the storage layer at the bottom, to the Array API layer at the top:

![Five layer diagram](images/design.svg)

Blue blocks are implemented in Cubed, green in Rechunker, and red in other projects like Zarr and Beam.

Let's go through the layers from the bottom:

## Storage

Every _array_ in Cubed is backed by a Zarr array. This means that the array type inherits Zarr attributes including the underlying store (which may be on local disk, or a cloud store, for example), as well as the shape, dtype, and chunks. Chunks are the unit of storage and computation in this system.

## Runtime

Cubed uses external runtimes for computation. It follows the Rechunker model (and uses its API) to delegate tasks to stateless executors, which include Python (in-process), Lithops, Modal, Beam, and other Rechunker executors like Dask and Prefect.


## Primitive operations

There are two primitive operations on arrays:

<dl>
  <dt><code>blockwise</code></dt>
  <dd>Applies a function to multiple blocks from multiple inputs, expressed using concise indexing rules.</dd>
  <dt><code>rechunk</code></dt>
  <dd>Changes the chunking of an array, without changing its shape or dtype.</dd>
</dl>

## Core operations

These are built on top of the primitive operations, and provide functions that are needed to implement all array operations.

<dl>
  <dt><code>elemwise</code></dt>
  <dd>Applies a function elementwise to its arguments, respecting broadcasting.</dd>
  <dt><code>map_blocks</code></dt>
  <dd>Applies a function to corresponding blocks from multiple inputs.</dd>
  <dt><code>map_direct</code></dt>
  <dd>Applies a function across blocks of a new array, reading directly from side inputs (not necessarily in a blockwise fashion).</dd>
  <dt><code>index</code> (<code>__getitem__</code>)</dt>
  <dd>Subsets an array, along one or more axes.</dd>
  <dt><code>reduction</code></dt>
  <dd>Applies a function to reduce an array along one or more axes.</dd>
  <dt><code>arg_reduction</code></dt>
  <dd>A reduction that returns the array indexes, not the values.</dd>
</dl>

## Array API

The new [Python Array API](https://data-apis.org/array-api/latest/) was chosen for the public API as it provides a useful, well-defined subset of the NumPy API. There are a few extensions, including Zarr IO, random number generation, and operations like `map_blocks` which are heavily used in Dask applications.
