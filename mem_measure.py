import numpy as np
import psutil
import zarr

# Simple test to see the effect of various numpy operations on memory usage
# See https://github.com/dask/distributed/issues/1409 for useful background


def get_mem():
    return psutil.Process().memory_info().rss


def numpy_array():
    print("Numpy array")

    mem0 = get_mem()

    # A numpy array has predictable memory usage
    # Note: don't use empty or zeros as it doesn't allocate memory in the same way
    x = np.ones(int(1e9), dtype="u1")
    print("Numpy nbytes", x.nbytes)

    mem1 = get_mem()
    print("Memory used after allocation", mem1 - mem0)

    del x

    mem2 = get_mem()
    print("Memory used after deletion", mem2 - mem0)

    print()


def numpy_transpose():
    print("Numpy transpose (reuse array)")

    mem0 = get_mem()

    x = np.ones(int(1e9), dtype="u1")
    print("Numpy nbytes", x.nbytes)

    mem1 = get_mem()
    print("Memory used after allocation", mem1 - mem0)

    y = np.transpose(x)

    mem2 = get_mem()
    print("Memory used after transpose", mem2 - mem0)

    print()


def numpy_negative():
    print("Numpy negative (new array)")

    mem0 = get_mem()

    x = np.ones(int(1e9), dtype="u1")
    print("Numpy nbytes", x.nbytes)

    mem1 = get_mem()
    print("Memory used after allocation", mem1 - mem0)

    y = np.negative(x)

    mem2 = get_mem()
    print("Memory used after negative", mem2 - mem0)

    print()


def zarr_storage():
    print("Zarr storage")

    mem0 = get_mem()

    # Copying a zarr array chunk by chunk only uses the chunk size of memory

    # Create a zarr array on disk
    store = zarr.TempStore()
    zarr.ones((1000, 1000, 1000), store=store, dtype="u1", chunks=(500, 500, 500))
    z = zarr.open(store)
    print("Zarr nbytes", z.nbytes)

    store2 = zarr.TempStore()
    zarr.empty((1000, 1000, 1000), store=store2, dtype="u1", chunks=(500, 500, 500))
    z2 = zarr.open(store2)

    z2[0:500, 0:500, 0:500] = z[0:500, 0:500, 0:500]
    z2[500:1000, 0:500, 0:500] = z[500:1000, 0:500, 0:500]
    z2[0:500, 500:1000, 0:500] = z[0:500, 500:1000, 0:500]
    z2[0:500, 0:500, 500:1000] = z[0:500, 0:500, 500:1000]

    mem1 = get_mem()
    print("Memory used after copying zarr chunks", mem1 - mem0)

    print()


if __name__ == "__main__":

    numpy_array()
    numpy_transpose()
    numpy_negative()
    zarr_storage()
