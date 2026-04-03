# Contributing

Contributions to Cubed are very welcome. Please head over to [GitHub](https://github.com/cubed-dev/cubed) to get involved.

## Development



### Using conda/pip

Create an environment with:

```shell
conda create --name cubed python=3.11
conda activate cubed
pip install -e ".[test]"
```

### Using uv

Install the project with test dependencies:

```shell
uv sync --extra test
```

Run the tests:

```shell
uv run pytest
```

To include additional extras (e.g. Dask support), add more `--extra` flags:

```shell
uv sync --extra test --extra dask
```

> **Note:** Do not use `--all-extras`, as it includes backend-specific extras
> (Beam, Lithops, Modal, Coiled) that have conflicting or platform-specific
> dependencies and are not needed for local development.

### Additional dependencies

Make sure `graphviz` is installed on your machine (see [these instructions](https://graphviz.org/download/)).
