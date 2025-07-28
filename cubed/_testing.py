import functools
import importlib.util

import numpy as np
import numpy.testing as npt

import cubed


@functools.cache
def has_cupy() -> bool:
    return importlib.util.find_spec("cupy") is not None


@functools.singledispatch
def to_numpy(a):
    return np.asarray(a)


@to_numpy.register(cubed.Array)
def _(a: cubed.Array) -> np.ndarray:
    return to_numpy(a.compute())


if has_cupy():
    import cupy

    @to_numpy.register(cupy.ndarray)
    def _(a):
        return a.get()


def assert_array_equal(a, b, **kwargs):
    a = to_numpy(a)
    b = to_numpy(b)
    npt.assert_array_equal(a, b, **kwargs)


def assert_allclose(a, b, **kwargs):
    a = to_numpy(a)
    b = to_numpy(b)
    npt.assert_allclose(a, b, **kwargs)
