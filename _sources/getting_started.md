# Getting Started

## Installation

### Conda

You can install cubed with a minimal set of dependencies using conda:

```shell
conda install -c conda-forge cubed
```

### Pip

You can also install cubed with pip:

```shell
python -m pip install cubed
```

Cubed has many optional dependencies, which can be installed in sets for different functionality (especially for running on different executors):

    $ python -m pip install "cubed[diagnostics]"  # Install optional dependencies for cubed diagnostics
    $ python -m pip install "cubed[beam]"         # Install optional dependencies for the beam executor
    $ python -m pip install "cubed[lithops]"      # Install optional dependencies for the lithops executor
    $ python -m pip install "cubed[modal]"        # Install optional dependencies for the modal executor

To see the full list of which packages are installed with which options see `[project.optional_dependencies]` in `pyproject.toml`:
```{eval-rst}
.. literalinclude:: ../pyproject.toml
   :language: ini
   :start-at: [project.optional-dependencies]
   :end-before: [project.urls]
```

## Example

Start with a simple example that runs using the local Python executor.

```python
>>> import cubed.array_api as xp
>>> import cubed.random
>>> spec = cubed.Spec(work_dir="tmp", allowed_mem=100_000)
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
