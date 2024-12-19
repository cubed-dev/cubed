## Array API Coverage Implementation Status

Cubed supports version [2023.12](https://data-apis.org/array-api/2023.12/index.html) of the Python array API standard, with a few exceptions noted below. The [Fourier transform functions](https://data-apis.org/array-api/2023.12/extensions/fourier_transform_functions.html) are *not* supported.

This table shows which parts of the the [Array API](https://data-apis.org/array-api/latest/API_specification/index.html) have been implemented in Cubed, and which ones are missing. The version column shows the version when the feature was added to the standard, for version 2022.12 or later.

| Category                 | Object/Function     | Implemented        | Version    | Notes                        |
| ------------------------ | ------------------- | ------------------ | ---------- | ---------------------------- |
| Array object             | Arithmetic Ops      | :white_check_mark: |            |                              |
|                          | Array Ops           | :white_check_mark: |            |                              |
|                          | Bitwise Ops         | :white_check_mark: |            |                              |
|                          | Comparison Ops      | :white_check_mark: |            |                              |
|                          | In-place Ops        | :x:                |            | Arrays are immutable         |
|                          | Reflected Ops       | :white_check_mark: |            |                              |
|                          | Attributes          | :white_check_mark: |            |                              |
|                          | Methods             | :white_check_mark: |            | Not device methods           |
| Constants                | `e`, `inf`, ...     | :white_check_mark: |            |                              |
| Creation Functions       | `arange`            | :white_check_mark: |            |                              |
|                          | `asarray`           | :white_check_mark: |            |                              |
|                          | `empty`             | :white_check_mark: |            |                              |
|                          | `empty_like`        | :white_check_mark: |            |                              |
|                          | `eye`               | :white_check_mark: |            |                              |
|                          | `from_dlpack`       | :x:                |            |                              |
|                          | `full`              | :white_check_mark: |            |                              |
|                          | `full_like`         | :white_check_mark: |            |                              |
|                          | `linspace`          | :white_check_mark: |            |                              |
|                          | `meshgrid`          | :white_check_mark: |            |                              |
|                          | `ones`              | :white_check_mark: |            |                              |
|                          | `ones_like`         | :white_check_mark: |            |                              |
|                          | `tril`              | :white_check_mark: |            |                              |
|                          | `triu`              | :white_check_mark: |            |                              |
|                          | `zeros`             | :white_check_mark: |            |                              |
|                          | `zeros_like`        | :white_check_mark: |            |                              |
| Data Type Functions      | `astype`            | :white_check_mark: |            |                              |
|                          | `can_cast`          | :white_check_mark: |            |                              |
|                          | `finfo`             | :white_check_mark: |            |                              |
|                          | `iinfo`             | :white_check_mark: |            |                              |
|                          | `result_type`       | :white_check_mark: |            |                              |
| Data Types               | `bool`, `int8`, ... | :white_check_mark: |            |                              |
| Elementwise Functions    | `add`               | :white_check_mark: |            | Example of a binary function |
|                          | `negative`          | :white_check_mark: |            | Example of a unary function  |
|                          | _others_            | :white_check_mark: |            |                              |
| Indexing                 | Single-axis         | :white_check_mark: |            |                              |
|                          | Multi-axis          | :white_check_mark: |            |                              |
|                          | Boolean array       | :x:                |            | Shape is data dependent, [#73](https://github.com/cubed-dev/cubed/issues/73) |
| Indexing Functions       | `take`              | :white_check_mark: | 2022.12    |                              |
| Inspection               | `capabilities`      | :white_check_mark: | 2023.12    |                              |
|                          | `default_device`    | :white_check_mark: | 2023.12    |                              |
|                          | `default_dtypes`    | :white_check_mark: | 2023.12    |                              |
|                          | `devices`           | :white_check_mark: | 2023.12    |                              |
|                          | `dtypes`            | :white_check_mark: | 2023.12    |                              |
| Linear Algebra Functions | `matmul`            | :white_check_mark: |            |                              |
|                          | `matrix_transpose`  | :white_check_mark: |            |                              |
|                          | `tensordot`         | :white_check_mark: |            |                              |
|                          | `vecdot`            | :white_check_mark: |            |                              |
| Manipulation Functions   | `broadcast_arrays`  | :white_check_mark: |            |                              |
|                          | `broadcast_to`      | :white_check_mark: |            |                              |
|                          | `concat`            | :white_check_mark: |            |                              |
|                          | `expand_dims`       | :white_check_mark: |            |                              |
|                          | `flip`              | :white_check_mark: |            |                              |
|                          | `permute_dims`      | :white_check_mark: |            |                              |
|                          | `repeat`            | :white_check_mark: | 2023.12    |                              |
|                          | `reshape`           | :white_check_mark: |            | Partial implementation       |
|                          | `roll`              | :white_check_mark: |            |                              |
|                          | `squeeze`           | :white_check_mark: |            |                              |
|                          | `stack`             | :white_check_mark: |            |                              |
|                          | `tile`              | :white_check_mark: | 2023.12    |                              |
|                          | `unstack`           | :white_check_mark: | 2023.12    |                              |
| Searching Functions      | `argmax`            | :white_check_mark: |            |                              |
|                          | `argmin`            | :white_check_mark: |            |                              |
|                          | `nonzero`           | :x:                |            | Shape is data dependent      |
|                          | `searchsorted`      | :x:                | 2023.12    |                              |
|                          | `where`             | :white_check_mark: |            |                              |
| Set Functions            | `unique_all`        | :x:                |            | Shape is data dependent      |
|                          | `unique_counts`     | :x:                |            | Shape is data dependent      |
|                          | `unique_inverse`    | :x:                |            | Shape is data dependent      |
|                          | `unique_values`     | :x:                |            | Shape is data dependent      |
| Sorting Functions        | `argsort`           | :x:                |            | Not in Dask                  |
|                          | `sort`              | :x:                |            | Not in Dask                  |
| Statistical Functions    | `cumulative_sum`    | :x:                | 2023.12    | WIP [#531](https://github.com/cubed-dev/cubed/pull/531) |
|                          | `max`               | :white_check_mark: |            |                              |
|                          | `mean`              | :white_check_mark: |            |                              |
|                          | `min`               | :white_check_mark: |            |                              |
|                          | `prod`              | :white_check_mark: |            |                              |
|                          | `std`               | :white_check_mark: |            |                              |
|                          | `sum`               | :white_check_mark: |            |                              |
|                          | `var`               | :white_check_mark: |            |                              |
| Utility Functions        | `all`               | :white_check_mark: |            |                              |
|                          | `any`               | :white_check_mark: |            |                              |

### Linear Algebra Extension

A few of the [linear algebra extension](https://data-apis.org/array-api/2022.12/extensions/linear_algebra_functions.html) functions are supported, as indicated in this table.

| Category                 | Object/Function     | Implemented        | Version    | Notes                        |
| ------------------------ | ------------------- | ------------------ | ---------- | ---------------------------- |
| Linear Algebra Functions | `cholesky`          | :x:                |            |                              |
|                          | `cross`             | :x:                |            |                              |
|                          | `det`               | :x:                |            |                              |
|                          | `diagonal`          | :x:                |            |                              |
|                          | `eigh`              | :x:                |            |                              |
|                          | `eigvalsh`          | :x:                |            |                              |
|                          | `inv`               | :x:                |            |                              |
|                          | `matmul`            | :white_check_mark: |            |                              |
|                          | `matrix_norm`       | :x:                |            |                              |
|                          | `matrix_power`      | :x:                |            |                              |
|                          | `matrix_rank`       | :x:                |            |                              |
|                          | `matrix_transpose`  | :white_check_mark: |            |                              |
|                          | `outer`             | :white_check_mark: |            |                              |
|                          | `pinv`              | :x:                |            |                              |
|                          | `qr`                | :white_check_mark: |            |                              |
|                          | `slogdet`           | :x:                |            |                              |
|                          | `solve`             | :x:                |            |                              |
|                          | `svd`               | :white_check_mark: |            |                              |
|                          | `svdvals`           | :white_check_mark: |            |                              |
|                          | `tensordot`         | :white_check_mark: |            |                              |
|                          | `trace`             | :x:                |            |                              |
|                          | `vecdot`            | :white_check_mark: |            |                              |
|                          | `vectornorm`        | :x:                |            |                              |
