# Getting Started

## Installation

Install Cubed with pip:

```shell
python -m pip install cubed
```

## Example

Start with a simple example that runs using the local Python executor.

```python
>>> import cubed.array_api as xp
>>> import cubed.random
>>> spec = cubed.Spec(work_dir="tmp", max_mem=100_000)
>>> a = cubed.random.random((4, 4), chunks=(2, 2), spec=spec)
>>> b = cubed.random.random((4, 4), chunks=(2, 2), spec=spec)
>>> c = xp.matmul(a, b)
>>> c.compute()
array([[1.22171031, 0.93644194, 1.83459119, 1.8087655 ],
       [1.3540541 , 1.13054495, 2.24504742, 2.05022751],
       [0.98211893, 0.62740696, 1.21686602, 1.26402294],
       [1.58566331, 1.33010476, 2.3994953 , 2.29258764]])
```

See more in the [demo notebook](https://github.com/tomwhite/cubed/blob/main/examples/demo.ipynb).

See the [examples README](https://github.com/tomwhite/cubed/tree/main/examples/README.md) for more about running on cloud services.
