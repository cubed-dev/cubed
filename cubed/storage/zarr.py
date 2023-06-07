import zarr


class LazyZarrArray:
    """A Zarr array that may not have been written to storage yet.

    On creation, a normal Zarr array's metadata is immediately written to storage,
    and there is no way to avoid this, hence this class, which separates the creation
    of the object in memory and the creation of array metadata in storage.
    """

    def __init__(
        self,
        shape,
        dtype,
        chunks,
        store,
        initial_values=None,
        fill_value=None,
        **kwargs,
    ):
        """Create a Zarr array lazily in memory."""
        # use an empty in-memory Zarr array as a template since it normalizes its properties
        template = zarr.empty(
            shape, dtype=dtype, chunks=chunks, store=zarr.storage.MemoryStore()
        )
        self.shape = template.shape
        self.dtype = template.dtype
        self.chunks = template.chunks

        self.store = store
        self.initial_values = initial_values
        self.fill_value = fill_value
        self.kwargs = kwargs

    def create(self, mode="w-"):
        """Create the Zarr array in storage.

        The Zarr array's metadata is initialized in the backing store, and any
        initial values are set on the array.

        Parameters
        ----------
        mode : str
            The mode to open the Zarr array with using ``zarr.open``.
            Default is 'w-', which means create, fail it already exists.
        """
        target = zarr.open(
            self.store,
            mode=mode,
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
            fill_value=self.fill_value,
            **self.kwargs,
        )
        if self.initial_values is not None and self.initial_values.size > 0:
            target[...] = self.initial_values
        return target

    def open(self):
        """Open the Zarr array for reading or writing and return it.

        Note that the Zarr array must have been created or this method will raise an exception.
        """
        # r+ means read/write, fail if it doesn't exist
        return zarr.open(
            self.store,
            mode="r+",
            shape=self.shape,
            dtype=self.dtype,
            chunks=self.chunks,
        )

    def __repr__(self):
        return f"cubed.storage.zarr.LazyZarrArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"


def lazy_empty(shape, *, dtype, chunks, store, **kwargs):
    return LazyZarrArray(shape, dtype, chunks, store, **kwargs)


def lazy_from_array(array, *, dtype, chunks, store, **kwargs):
    return LazyZarrArray(
        array.shape, dtype, chunks, store, initial_values=array, **kwargs
    )


def lazy_full(shape, fill_value, *, dtype, chunks, store, **kwargs):
    return LazyZarrArray(shape, dtype, chunks, store, fill_value=fill_value, **kwargs)


def open_if_lazy_zarr_array(array):
    """If array is a LazyZarrArray then open it, leaving other arrays unchanged."""
    return array.open() if isinstance(array, LazyZarrArray) else array
