# Cubed: Scalable array processing with bounded memory

Cubed is a Python library for scalable out-of-core multi-dimensional array processing with bounded memory.

::::{grid} 2
:gutter: 2

:::{grid-item-card}  Familiar API
Cubed provides NumPy and Xarray APIs for processing your multi-dimensional array data
:::
:::{grid-item-card}  Dask replacement
Cubed is a drop-in replacement for Dask's Array API
:::
:::{grid-item-card}  Predictable memory usage
Cubed will tell you if your computation would run out of memory *before* running it
:::
:::{grid-item-card}  Reliable
Cubed is designed to be robust to failures and will reliably complete a computation
:::
:::{grid-item-card}  Run locally
Cubed can process hundreds of GB of array data on your laptop using all available cores
:::
:::{grid-item-card}  Scale in the cloud
Cubed is horizontally scalable and stateless, and can scale to multi-TB datasets in the cloud
:::
::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: For users
getting-started
user-guide/index
examples/index
api
array-api
configuration
why-cubed
related-projects
articles
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: For developers
design
operations
computation
contributing
```
