# Contributing

Contributions to Cubed are very welcome. Please head over to [GitHub](https://github.com/tomwhite/cubed) to get involved.

## Development

Create an environment with

```shell
conda create --name cubed python=3.10
conda activate cubed
pip install -r requirements.txt
pip install -e .
```

Make sure `graphviz` is installed on your machine (see [these instructions](https://graphviz.org/download/)).

Optionally, to run Jax on the M1+ Mac, please follow these instructions from Apple:
https://developer.apple.com/metal/jax/

To summarize: 
```shell
pip install jax-metal
export CUBED_BACKEND_ARRAY_API_MODULE=jax.numpy
export JAX_ENABLE_X64=False
export CUBED_DEFAULT_PRECISION_X32=True
export ENABLE_PJRT_COMPATIBILITY=True
```

Please make sure that your version of Python and all dependencies are compiled for ARM.