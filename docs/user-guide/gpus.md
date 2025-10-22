# GPU Support

Cubed has experimental support for using GPU-backed ndarrays. With zarr-python's
[native GPU support], you can load data into GPU memory, perform some Cubed
computation on the GPU, and write the result, while minimizing the number of host
to device transfers.

```{note}
Currently only NVIDIA GPUs and [CuPy] arrays are supported.
```

Set the following environment variables to control whether host or device arrays
are in Cubed and Zarr.

```shell
# syntax may differ in your shell
export CUBED_BACKEND_ARRAY_API_MODULE="array_api_compat.cupy"
export ZARR_BUFFER="zarr.buffer.gpu.Buffer"
export ZARR_NDBUFFER="zarr.buffer.gpu.NDBuffer"
```


[native GPU support]: https://zarr.readthedocs.io/en/stable/user-guide/gpu.html
[CuPy]: https://cupy.dev/
