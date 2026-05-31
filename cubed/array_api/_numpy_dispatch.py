import os

_HANDLED_FUNCTIONS = {}

CUBED_NUMPY_COMPAT = "CUBED_NUMPY_COMPAT" in os.environ


def implements(*numpy_functions):
    """Register a cubed implementation of one or more numpy functions."""

    def decorator(cubed_function):
        for numpy_function in numpy_functions:
            _HANDLED_FUNCTIONS[numpy_function] = cubed_function
        return cubed_function

    return decorator
