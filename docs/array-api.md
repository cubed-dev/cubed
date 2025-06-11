# Python Array API

Cubed implements version 2023.12 of the [Python Array API standard](https://data-apis.org/array-api/2023.12/index.html) in `cubed.array_api`, with a few exceptions listed on the [coverage status](https://github.com/cubed-dev/cubed/blob/main/api_status.md) page. The [Fourier transform functions](https://data-apis.org/array-api/2023.12/extensions/fourier_transform_functions.html) are *not* supported.

## Differences between Cubed and the standard

The following [Creation Functions](https://data-apis.org/array-api/latest/API_specification/creation_functions.html) accept extra `chunks` and `spec` keyword arguments:

```{eval-rst}
.. autofunction:: cubed.array_api.arange
.. autofunction:: cubed.array_api.asarray
.. autofunction:: cubed.array_api.empty
.. autofunction:: cubed.array_api.empty_like
.. autofunction:: cubed.array_api.eye
.. autofunction:: cubed.array_api.full
.. autofunction:: cubed.array_api.full_like
.. autofunction:: cubed.array_api.linspace
.. autofunction:: cubed.array_api.ones
.. autofunction:: cubed.array_api.ones_like
.. autofunction:: cubed.array_api.zeros
.. autofunction:: cubed.array_api.zeros_like
```

The following [Manipulation Functions](https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html) accept extra `chunks` keyword arguments:

```{eval-rst}
.. autofunction:: cubed.array_api.broadcast_to
.. autofunction:: cubed.array_api.concat
```
