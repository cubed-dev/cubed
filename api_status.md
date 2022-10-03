## Array API Coverage Implementation Status

This table shows which parts of the the [Array API](https://data-apis.org/array-api/latest/API_specification/index.html) have been implemented in Cubed. For those that have not been implemented a rough level of difficulty is indicated (1=easy, 3=hard).

| Category                 | Object/Function     | Implemented        | Difficulty | Notes                        |
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
|                          | `can_cast`          | :white_check_mark: |            | Same as `numpy.array_api`    |
|                          | `finfo`             | :white_check_mark: |            | Same as `numpy.array_api`    |
|                          | `iinfo`             | :white_check_mark: |            | Same as `numpy.array_api`    |
|                          | `result_type`       | :white_check_mark: |            | Same as `numpy.array_api`    |
| Data Types               | `bool`, `int8`, ... | :white_check_mark: |            |                              |
| Elementwise Functions    | `add`               | :white_check_mark: |            | Example of a binary function |
|                          | `negative`          | :white_check_mark: |            | Example of a unary function  |
|                          | _others_            | :white_check_mark: |            |                              |
| Indexing                 | Single-axis         | :white_check_mark: |            |                              |
|                          | Multi-axis          | :white_check_mark: |            |                              |
|                          | Boolean array       | :x:                | 3          | Shape is data dependent, [#73](https://github.com/tomwhite/cubed/issues/73) |
| Linear Algebra Functions | `matmul`            | :white_check_mark: |            |                              |
|                          | `matrix_transpose`  | :white_check_mark: |            |                              |
|                          | `tensordot`         | :white_check_mark: |            |                              |
|                          | `vecdot`            | :white_check_mark: |            |                              |
| Manipulation Functions   | `broadcast_arrays`  | :white_check_mark: |            |                              |
|                          | `broadcast_to`      | :white_check_mark: |            |                              |
|                          | `concat`            | :white_check_mark: |            |                              |
|                          | `expand_dims`       | :white_check_mark: |            |                              |
|                          | `flip`              | :x:                | 2          | Needs indexing with step=-1, [#114](https://github.com/tomwhite/cubed/issues/114) |
|                          | `permute_dims`      | :white_check_mark: |            |                              |
|                          | `reshape`           | :white_check_mark: |            | Partial implementation       |
|                          | `roll`              | :x:                | 2          | Use `concat` and `reshape`, [#115](https://github.com/tomwhite/cubed/issues/115) |
|                          | `squeeze`           | :white_check_mark: |            |                              |
|                          | `stack`             | :white_check_mark: |            |                              |
| Searching Functions      | `argmax`            | :white_check_mark: |            |                              |
|                          | `argmin`            | :white_check_mark: |            |                              |
|                          | `nonzero`           | :x:                | 3          | Shape is data dependent      |
|                          | `where`             | :white_check_mark: |            |                              |
| Set Functions            | `unique_all`        | :x:                | 3          | Shape is data dependent      |
|                          | `unique_counts`     | :x:                | 3          | Shape is data dependent      |
|                          | `unique_inverse`    | :x:                | 3          | Shape is data dependent      |
|                          | `unique_values`     | :x:                | 3          | Shape is data dependent      |
| Sorting Functions        | `argsort`           | :x:                | 3          | Not in Dask                  |
|                          | `sort`              | :x:                | 3          | Not in Dask                  |
| Statistical Functions    | `max`               | :white_check_mark: |            |                              |
|                          | `mean`              | :white_check_mark: |            |                              |
|                          | `min`               | :white_check_mark: |            |                              |
|                          | `prod`              | :white_check_mark: |            |                              |
|                          | `std`               | :x:                | 2          | Like `mean`, [#29](https://github.com/tomwhite/cubed/issues/29) |
|                          | `sum`               | :white_check_mark: |            |                              |
|                          | `var`               | :x:                | 2          | Like `mean`, [#29](https://github.com/tomwhite/cubed/issues/29) |
| Utility Functions        | `all`               | :white_check_mark: |            |                              |
|                          | `any`               | :white_check_mark: |            |                              |
