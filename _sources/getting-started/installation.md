# Installation

## Conda

You can install Cubed with a minimal set of dependencies using conda:

```shell
conda install -c conda-forge cubed
```

## Pip

You can also install Cubed with pip:

```shell
python -m pip install cubed
```

## Optional dependencies

Cubed has many optional dependencies, which can be installed in sets for different functionality (especially for running on different executors):

    $ python -m pip install "cubed[diagnostics]"  # Install optional dependencies for cubed diagnostics
    $ python -m pip install "cubed[beam]"         # Install optional dependencies for the beam executor
    $ python -m pip install "cubed[lithops]"      # Install optional dependencies for the lithops executor
    $ python -m pip install "cubed[modal]"        # Install optional dependencies for the modal executor

To see the full list of which packages are installed with which options see `[project.optional_dependencies]` in `pyproject.toml`:
```{eval-rst}
.. literalinclude:: ../../pyproject.toml
   :language: ini
   :start-at: [project.optional-dependencies]
   :end-before: [project.urls]
```
