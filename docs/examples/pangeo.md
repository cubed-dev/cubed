# Pangeo

## Notebooks

The following example notebooks demonstrate the use of Cubed with Xarray to tackle some challenging Pangeo workloads:

1. [Pangeo Vorticity Workload](https://github.com/cubed-dev/cubed/blob/main/examples/pangeo-1-vorticity.ipynb)
2. [Pangeo Quadratic Means Workload](https://github.com/cubed-dev/cubed/blob/main/examples/pangeo-2-quadratic-means.ipynb)
3. [Pangeo Transformed Eulerian Mean Workload](https://github.com/cubed-dev/cubed/blob/main/examples/pangeo-3-tem.ipynb)
4. [Pangeo Climatological Anomalies Workload](https://github.com/cubed-dev/cubed/blob/main/examples/pangeo-4-climatological-anomalies.ipynb)

## Running the notebook examples

Before running these notebook examples, you will need to install some additional dependencies (besides Cubed).

`conda install rich pydot flox cubed-xarray`

`cubed-xarray` is necessary to wrap Cubed arrays as Xarray DataArrays or Xarray Datasets.
`flox` is for supporting efficient groupby operations in Xarray.
`pydot` allows plotting the Cubed execution plan.
`rich` is for showing progress of array operations within callbacks applied to Cubed plan operations.
