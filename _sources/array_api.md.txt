# Python Array API

Cubed implements the [Python Array API standard](https://data-apis.org/array-api/latest/) in `cubed.array_api`. Refer
to its [API specification](https://data-apis.org/array-api/latest/API_specification/index.html) for API documentation.

## Missing from Cubed

The following parts of the standard are not implemented:

| Category               | Object/Function  |
| ---------------------- | ---------------- |
| Array object           | In-place Ops     |
| Creation Functions     | `from_dlpack`    |
| Indexing               | Boolean array    |
| Manipulation Functions | `flip`           |
|                        | `roll`           |
| Searching Functions    | `nonzero`        |
| Set Functions          | `unique_all`     |
|                        | `unique_counts`  |
|                        | `unique_inverse` |
|                        | `unique_values`  |
| Sorting Functions      | `argsort`        |
|                        | `sort`           |
| Statistical Functions  | `std`            |
|                        | `var`            |

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
```
