import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import cubed
import cubed.array_api as xp
from cubed import apply_gufunc
from cubed.backend_array_api import namespace as nxp


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=1000000)


@pytest.mark.parametrize("vectorize", [False, True])
def test_apply_reduction(spec, vectorize):
    def stats(x):
        # note dtype matches output_dtypes in apply_gufunc below
        return nxp.mean(x, axis=-1, dtype=np.float32)

    r = np.random.normal(size=(10, 20, 30))
    a = cubed.from_array(r, chunks=(5, 5, 30), spec=spec)
    actual = apply_gufunc(stats, "(i)->()", a, output_dtypes="f", vectorize=vectorize)
    expected = nxp.mean(r, axis=-1, dtype=np.float32)

    assert actual.compute().shape == expected.shape
    assert_allclose(actual.compute(), expected)


def test_apply_gufunc_elemwise_01(spec):
    def add(x, y):
        return x + y

    a = cubed.from_array(np.array([1, 2, 3]), chunks=2, spec=spec)
    b = cubed.from_array(np.array([1, 2, 3]), chunks=2, spec=spec)
    z = apply_gufunc(add, "(),()->()", a, b, output_dtypes=a.dtype)
    assert_equal(z, np.array([2, 4, 6]))


def test_apply_gufunc_elemwise_01_non_cubed_input(spec):
    def add(x, y):
        return x + y

    a = cubed.from_array(np.array([1, 2, 3]), chunks=3, spec=spec)
    b = np.array([1, 2, 3])
    z = apply_gufunc(add, "(),()->()", a, b, output_dtypes=a.dtype)
    assert_equal(z, np.array([2, 4, 6]))


def test_apply_gufunc_elemwise_loop(spec):
    def foo(x):
        assert x.shape in ((2,), (1,))
        return 2 * x

    a = cubed.from_array(np.array([1, 2, 3]), chunks=2, spec=spec)
    z = apply_gufunc(foo, "()->()", a, output_dtypes=int)
    assert z.chunks == ((2, 1),)
    assert_equal(z, np.array([2, 4, 6]))


def test_apply_gufunc_elemwise_core(spec):
    def foo(x):
        assert x.shape == (3,)
        return 2 * x

    a = cubed.from_array(np.array([1, 2, 3]), chunks=3, spec=spec)
    z = apply_gufunc(foo, "(i)->(i)", a, output_dtypes=int)
    assert z.chunks == ((3,),)
    assert_equal(z, np.array([2, 4, 6]))


def test_gufunc_two_inputs(spec):
    def foo(x, y):
        return np.einsum("...ij,...jk->ik", x, y)

    a = xp.ones((2, 3), chunks=100, dtype=int, spec=spec)
    b = xp.ones((3, 4), chunks=100, dtype=int, spec=spec)
    x = apply_gufunc(foo, "(i,j),(j,k)->(i,k)", a, b, output_dtypes=int)
    assert_equal(x, 3 * np.ones((2, 4), dtype=int))


def test_apply_gufunc_axes_two_kept_coredims(spec):
    ra = np.random.normal(size=(20, 30))
    rb = np.random.normal(size=(10, 1, 40))

    a = cubed.from_array(ra, chunks=(10, 30), spec=spec)
    b = cubed.from_array(rb, chunks=(5, 1, 40), spec=spec)

    def outer_product(x, y):
        return np.einsum("i,j->ij", x, y)

    c = apply_gufunc(outer_product, "(i),(j)->(i,j)", a, b, vectorize=True)
    assert c.compute().shape == (10, 20, 30, 40)


def test_gufunc_output_sizes(spec):
    def foo(x):
        return nxp.broadcast_to(x[:, np.newaxis], (x.shape[0], 3))

    a = cubed.from_array(np.array([1, 2, 3, 4, 5], dtype=int), spec=spec)
    x = apply_gufunc(foo, "()->(i_0)", a, output_dtypes=int, output_sizes={"i_0": 3})
    assert x.chunks == ((5,), (3,))
    assert_equal(
        x,
        np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ]
        ),
    )
