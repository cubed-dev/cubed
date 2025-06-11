# Cubed

## Scalable array processing with bounded memory

Cubed is a Python library for scalable out-of-core multi-dimensional array processing with bounded memory.

- Cubed provides NumPy and Xarray APIs for processing your multi-dimensional array data
- Cubed is a drop-in replacement for Dask's Array API
- Cubed will tell you if your computation would run out of memory *before* running it
- Cubed is designed to be robust to failures and will reliably complete a computation
- Cubed can process hundreds of GB of array data on your laptop using all available cores
- Cubed is horizontally scalable and stateless, and can scale to multi-TB datasets in the cloud

[Documentation](https://cubed-dev.github.io/cubed/)
