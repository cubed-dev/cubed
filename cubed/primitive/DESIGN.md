# Blockwise design

_Blockwise_ is a term from Dask that describes a class of functions operating on chunked arrays that map input chunks to output chunks.

Dask has a notation for describing the mapping (which Cubed supports), but the code here generalises this to support an arbitrary mapping.

## Model

* Chunked arrays have names and chunk coordinates.
* A `ChunkKey` refers to a chunk in an array by name and (a single) coordinate.
* A blockwise _function_ takes a set of input chunks (possibly from multiple arrays) and produces a set of output chunks (possibly for multiple arrays).
* A _back key function_ maps a `ChunkKey` in an output array to the input `ChunkKey`s needed for the blockwise _function_.
* A `BlockwiseOp` has a _back key function_, a blockwise _function_, and the names of the output arrays.

The _back key function_ returns an object that matches the structure of blockwise function, and encodes information about the access pattern for the input chunks.

In particular, a _list_ of `ChunkKey`s indicates that the chunks can be read all at once (e.g. in parallel), whereas an _iterator_ of `ChunkKey`s indicates that the chunks must be read one at a time. This is important for memory management.

### Example: a simple elementwise operation

Consider a simple operation that negates the input. Call the input array `a` and the output `b`.

* The _back key function_ takes a `ChunkKey` and returns a `ChunkKey` with the same coordinates but with a different name (`b` in this case).
* The _function_ takes a single value (chunk) and returns the negated value (chunk). (The chunk is an array object itself.)

## Application

To use a `BlockwiseOp` to compute output arrays given input arrays, follow this procedure:

* For each chunk in the output arrays, create a `ChunkKey`:
    * Call the _back key function_ with the `ChunkKey` to find the input `ChunkKey`s.
    * Read the chunks from the input arrays referred to by the `ChunkKey`s
    * Call the blockwise _function_ with the chunks to compute the output chunks
    * Write the output chunks to the output

Note that this procedure can be run in parallel across output chunks.

More generally, a computation may be defined as a DAG, where nodes are arrays or operations (`BlockwiseOp`) and edges are between arrays and operations.
The above procedure is applied to operations in the DAG in topologically sorted order, so that arrays that are inputs to other arrays are computed before the other arrays.

## Fusion

A `BlockwiseOp` operation can be _fused_ with its predecessor operations by separately fusing the back key functions and the blockwise functions.
The DAG can be re-written to replace the operations being fused with the single fused operation.

By repeatedly fusing operations in the DAG, it is reduced in size, which helps reduce the amount of IO, since the number of read/write array operations is reduced. However, depending on the nature of the blockwise operations, fusion may increase memory usage, so it must be applied judiciously.
