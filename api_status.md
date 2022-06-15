## Array API Coverage Implementation Status

This table shows which [Array API functions](https://data-apis.org/array-api/latest/API_specification/index.html) have been implemented in Cubed. For those that have not been implemented a rough level of difficulty is indicated (1=easy, 3=hard).

| Category       | Function           | Implemented        | Difficulty (1-3) | Notes                        |
| -------------- | ------------------ | ------------------ | ---------------- | ---------------------------- |
| Creation       | `arange`           | :white_check_mark: |                  | Partial implementation       |
|                | `asarray`          | :white_check_mark: |                  |                              |
|                | `empty`            | :white_check_mark: |                  | Uses `full`                  |
|                | `empty_like`       | :white_check_mark: |                  |                              |
|                | `eye`              |                    |                  |                              |
|                | `from_dlpack`      |                    |                  |                              |
|                | `full`             | :white_check_mark: |                  |                              |
|                | `full_like`        | :white_check_mark: |                  |                              |
|                | `linspace`         |                    | 2                | Like `arange`                |
|                | `meshgrid`         |                    |                  |                              |
|                | `ones`             | :white_check_mark: |                  | Uses `full`                  |
|                | `ones_like`        | :white_check_mark: |                  |                              |
|                | `tril`             |                    |                  |                              |
|                | `triu`             |                    |                  |                              |
|                | `zeros`            | :white_check_mark: |                  | Uses `full`                  |
|                | `zeros_like`       | :white_check_mark: |                  |                              |
| Data Type      | `astype`           | :white_check_mark: |                  |                              |
|                | `can_cast`         | :white_check_mark: |                  | Same as `numpy.array_api`    |
|                | `finfo`            | :white_check_mark: |                  | Same as `numpy.array_api`    |
|                | `iinfo`            | :white_check_mark: |                  | Same as `numpy.array_api`    |
|                | `result_type`      | :white_check_mark: |                  | Same as `numpy.array_api`    |
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
|                | `concat`           |                    | 3                | Like `stack`                 |
|                | `expand_dims`      | :white_check_mark: |                  |                              |
|                | `flip`             |                    | 3                | Needs indexing               |
|                | `permute_dims`     | :white_check_mark: |                  |                              |
|                | `reshape`          | :white_check_mark: |                  | Partial implementation       |
|                | `roll`             |                    | 3                | Needs `concat` and `reshape` |
|                | `squeeze`          | :white_check_mark: |                  |                              |
|                | `stack`            | :white_check_mark: |                  |                              |
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
|                | `any`              | :white_check_mark: |                  |                              |

TODO: add other parts of the spec (Array object, for example) and their level of coverage.
