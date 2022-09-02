import pytest

import numpy as np
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed import apply_gufunc
from cubed.tests.utils import ALL_EXECUTORS, MAIN_EXECUTORS, MODAL_EXECUTORS



@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000)


@pytest.fixture(scope="module", params=MAIN_EXECUTORS)
def executor(request):
    return request.param


@pytest.fixture(scope="module", params=MODAL_EXECUTORS)
def modal_executor(request):
    return request.param


# TODO vectorize
def test_apply_reduction(spec):
    def stats(x):
        return np.mean(x, axis=-1)

    r = np.random.normal(size=(10, 20, 30))
    a = cubed.from_array(r, chunks=(5, 5, 30), spec=spec)
    actual = apply_gufunc(stats, "(i)->()", a, output_dtypes="f", vectorize=False)
    expected = stats(r)

    assert actual.compute().shape == expected.shape
    assert_array_equal(actual.compute(), expected)


@pytest.mark.xfail(reason="Not implemented - blockwise doesn't yet support multiple outputs")
def test_apply_two_outputs(spec):
    def stats(x):
        return np.mean(x, axis=-1), np.std(x, axis=-1)

    r = np.random.normal(size=(10, 20, 30))
    a = cubed.from_array(r, chunks=(5, 5, 30), spec=spec)
    result = apply_gufunc(stats, "(i)->(),()", a, output_dtypes=2 * (a.dtype,))
    expected_mean, expected_std = stats(r)

    assert isinstance(result, tuple)
    mean, std = result
    assert actual_mean.compute().shape == expected_mean
    assert_array_equal(actual_mean.compute(), expected_mean)
    assert actual_std.compute().shape == expected_std
    assert_array_equal(actual_std.compute(), expected_std)
