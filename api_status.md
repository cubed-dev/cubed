## Array API Coverage Implementation Status

This table shows which parts of the the [Array API](https://data-apis.org/array-api/latest/API_specification/index.html) have been implemented in Cubed. For those that have not been implemented a rough level of difficulty is indicated (1=easy, 3=hard).

| Category                 | Object/Function     | Implemented        | Difficulty | Notes                         |
| ------------------------ | ------------------- | ------------------ | ---------- | ----------------------------- |
| Array object             | Arithmetic Ops      | :white_check_mark: |            |                               |
|                          | Array Ops           |                    |            |                               |
|                          | Bitwise Ops         |                    |            |                               |
|                          | Comparison Ops      | :white_check_mark: |            |                               |
|                          | In-place Ops        |                    |            |                               |
|                          | Reflected Ops       |                    |            |                               |
|                          | Attributes          | :white_check_mark: |            |                               |
|                          | Methods             |                    |            | Partial implementation        |
| Constants                | `e`, `inf`, ...     | :white_check_mark: |            |                               |
| Creation Functions       | `arange`            | :white_check_mark: |            |                               |
|                          | `asarray`           | :white_check_mark: |            |                               |
|                          | `empty`             | :white_check_mark: |            | Uses `full`                   |
|                          | `empty_like`        | :white_check_mark: |            |                               |
|                          | `eye`               |                    |            |                               |
|                          | `from_dlpack`       |                    |            |                               |
|                          | `full`              | :white_check_mark: |            |                               |
|                          | `full_like`         | :white_check_mark: |            |                               |
|                          | `linspace`          |                    | 2          | Like `arange`                 |
|                          | `meshgrid`          |                    |            |                               |
|                          | `ones`              | :white_check_mark: |            | Uses `full`                   |
|                          | `ones_like`         | :white_check_mark: |            |                               |
|                          | `tril`              |                    |            |                               |
|                          | `triu`              |                    |            |                               |
|                          | `zeros`             | :white_check_mark: |            | Uses `full`                   |
|                          | `zeros_like`        | :white_check_mark: |            |                               |
| Data Type Functions      | `astype`            | :white_check_mark: |            |                               |
|                          | `can_cast`          | :white_check_mark: |            | Same as `numpy.array_api`     |
|                          | `finfo`             | :white_check_mark: |            | Same as `numpy.array_api`     |
|                          | `iinfo`             | :white_check_mark: |            | Same as `numpy.array_api`     |
|                          | `result_type`       | :white_check_mark: |            | Same as `numpy.array_api`     |
| Data Types               | `bool`, `int8`, ... | :white_check_mark: |            |                               |
| Elementwise Functions    | `add`               | :white_check_mark: |            | Example of a binary function  |
|                          | `negative`          | :white_check_mark: |            | Example of a unary function   |
|                          | ...                 |                    |            |                               |
|                          | _others_            |                    | 1          | Like `add` or `negative`      |
| Indexing                 | Single-axis         | :white_check_mark: |            |                               |
|                          | Multi-axis          | :white_check_mark: |            | Can't mix integers and slices |
|                          | Boolean array       |                    | 3          | Shape is data dependent       |
| Linear Algebra Functions | `matmul`            | :white_check_mark: |            | Only 2D case                  |
|                          | `matrix_transpose`  | :white_check_mark: |            |                               |
|                          | `tensordot`         |                    | 2          | Like Dask                     |
|                          | `vecdot`            |                    | 1          | Express using `tensordot`     |
| Manipulation Functions   | `broadcast_arrays`  | :white_check_mark: |            |                               |
|                          | `broadcast_to`      | :white_check_mark: |            |                               |
|                          | `concat`            | :white_check_mark: |            |                               |
|                          | `expand_dims`       | :white_check_mark: |            |                               |
|                          | `flip`              |                    | 3          | Needs indexing                |
|                          | `permute_dims`      | :white_check_mark: |            |                               |
|                          | `reshape`           | :white_check_mark: |            | Partial implementation        |
|                          | `roll`              |                    | 3          | Needs `concat` and `reshape`  |
|                          | `squeeze`           | :white_check_mark: |            |                               |
|                          | `stack`             | :white_check_mark: |            |                               |
| Searching Functions      | `argmax`            | :white_check_mark: |            |                               |
|                          | `argmin`            | :white_check_mark: |            |                               |
|                          | `nonzero`           |                    | 3          | Shape is data dependent       |
|                          | `where`             | :white_check_mark: |            |                               |
| Set Functions            | `unique_all`        |                    | 3          | Shape is data dependent       |
|                          | `unique_counts`     |                    | 3          | Shape is data dependent       |
|                          | `unique_inverse`    |                    | 3          | Shape is data dependent       |
|                          | `unique_values`     |                    | 3          | Shape is data dependent       |
| Sorting Functions        | `argsort`           |                    | 3          | Not in Dask                   |
|                          | `sort`              |                    | 3          | Not in Dask                   |
| Statistical Functions    | `max`               | :white_check_mark: |            |                               |
|                          | `mean`              | :white_check_mark: |            |                               |
|                          | `min`               | :white_check_mark: |            |                               |
|                          | `prod`              | :white_check_mark: |            |                               |
|                          | `std`               |                    | 2          | Like `mean`                   |
|                          | `sum`               | :white_check_mark: |            |                               |
|                          | `var`               |                    | 2          | Like `mean`                   |
| Utility Functions        | `all`               | :white_check_mark: |            |                               |
|                          | `any`               | :white_check_mark: |            |                               |
