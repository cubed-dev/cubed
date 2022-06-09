## Array API Coverage Implementation Status

This table shows which [Array API functions](https://data-apis.org/array-api/latest/API_specification/index.html) have been implemented in Cubed. For those that have not been implemented a rough level of difficulty is indicated (1=easy, 3=hard).

| Category       | Function           | Implemented        | Difficulty (1-3) | Notes                        |
| -------------- | ------------------ | ------------------ | ---------------- | ---------------------------- |
| Creation       | `arange`           | :white_check_mark: |                  | Partial implementation       |
|                | `asarray`          | :white_check_mark: |                  |                              |
|                | `empty`            |                    | 1                | Like `ones`                  |
|                | `empty_like`       |                    | 1                | Like `ones`                  |
|                | `eye`              |                    |                  |                              |
|                | `from_dlpack`      |                    |                  |                              |
|                | `full`             |                    | 1                | Like `ones`                  |
|                | `full_like`        |                    | 1                | Like `ones`                  |
|                | `linspace`         |                    | 2                | Like `arange`                |
|                | `meshgrid`         |                    |                  |                              |
|                | `ones`             | :white_check_mark: |                  |                              |
|                | `ones_like`        |                    | 1                | Like `ones`                  |
|                | `tril`             |                    |                  |                              |
|                | `triu`             |                    |                  |                              |
|                | `zeros`            |                    | 1                | Like `ones`                  |
|                | `zeros_like`       |                    | 1                | Like `ones`                  |
| Data Type      | `astype`           | :white_check_mark: |                  |                              |
|                | `can_cast`         |                    | 1                | Same as `numpy.array_api`    |
|                | `finfo`            |                    | 1                | Same as `numpy.array_api`    |
|                | `iinfo`            |                    | 1                | Same as `numpy.array_api`    |
|                | `result_type`      | :white_check_mark: |                  |                              |
| Elementwise    | `add`              | :white_check_mark: |                  | Example of a binary function |
|                | `negative`         | :white_check_mark: |                  | Example of a unary function  |
|                | ...                |                    |                  |                              |
|                | _others_           |                    | 1                | Like `add` or `negative`     |
| Linear Algebra | `matmul`           | :white_check_mark: |                  | Only 2D case                 |
|                | `matrix_transpose` |                    | 1                | Like Dask                    |
|                | `tensordot`        |                    | 1                | Like Dask                    |
|                | `vecdot`           |                    | 1                | Express using `tensordot`    |
| Manipulation   | `broadcast_arrays` | :white_check_mark: |                  |                              |
|                | `broadcast_to`     | :white_check_mark: |                  |                              |
|                | `concat`           |                    | 3                | Primitive (Zarr view)        |
|                | `expand_dims`      |                    | 1                | Like `squeeze` or Zarr view  |
|                | `flip`             |                    | 3                | Needs indexing               |
|                | `permute_dims`     | :white_check_mark: |                  |                              |
|                | `reshape`          | :white_check_mark: | 3                | Partial implementation       |
|                | `roll`             |                    | 3                | Needs `concat` and `reshape` |
|                | `squeeze`          | :white_check_mark: |                  |                              |
|                | `stack`            |                    | 2                | Primitive (Zarr view)        |
| Searching      | `argmax`           |                    | 2                | `argreduction` primitive     |
|                | `argmin`           |                    | 2                | `argreduction` primitive     |
|                | `nonzero`          |                    | 3                | Shape is data dependent      |
|                | `where`            |                    | 1                |                              |
| Set            | `unique_all`       |                    | 3                | Shape is data dependent      |
|                | `unique_counts`    |                    | 3                | Shape is data dependent      |
|                | `unique_inverse`   |                    | 3                | Shape is data dependent      |
|                | `unique_values`    |                    | 3                | Shape is data dependent      |
| Sorting        | `argsort`          |                    | 3                | Not in Dask                  |
|                | `sort`             |                    | 3                | Not in Dask                  |
| Statistical    | `max`              |                    | 1                | Like `sum`                   |
|                | `mean`             | :white_check_mark: |                  |                              |
|                | `min`              |                    | 1                | Like `sum`                   |
|                | `prod`             |                    | 1                | Like `sum`                   |
|                | `std`              |                    | 2                | Like `mean`                  |
|                | `sum`              | :white_check_mark: |                  |                              |
|                | `var`              |                    | 2                | Like `mean`                  |
| Utility        | `all`              | :white_check_mark: |                  |                              |
|                | `any`              |                    | 1                | Like `all`                   |

TODO: add other parts of the spec (Array object, for example) and their level of coverage.
