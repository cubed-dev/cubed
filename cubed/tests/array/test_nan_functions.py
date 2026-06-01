import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

import cubed
import cubed.array_api as xp
from cubed._testing import assert_allclose, assert_array_equal
from cubed.backend_array_api import namespace as nxp

xps = make_strategies_namespace(nxp)


@pytest.fixture
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


@pytest.mark.parametrize(
    ("cubed_func", "numpy_func"),
    [
        (cubed.nanargmax, np.nanargmax),
        (cubed.nanargmin, np.nanargmin),
        (cubed.nancumprod, np.nancumprod),
        (cubed.nancumsum, np.nancumsum),
        (cubed.nanmax, np.nanmax),
        (cubed.nanmean, np.nanmean),
        (cubed.nanmin, np.nanmin),
        (cubed.nanprod, np.nanprod),
        (cubed.nanstd, np.nanstd),
        (cubed.nansum, np.nansum),
        (cubed.nanvar, np.nanvar),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_nan_function(spec, cubed_func, numpy_func, axis):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, xp.nan]], chunks=(2, 2), spec=spec)
    b = cubed_func(a, axis=axis)
    assert_array_equal(
        b.compute(),
        numpy_func(np.array([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]]), axis=axis),
    )


@pytest.mark.parametrize(
    ("cubed_func", "numpy_func"),
    [
        (cubed.nanargmax, np.nanargmax),
        (cubed.nanargmin, np.nanargmin),
    ],
)
def test_nan_function_allnan_error(spec, cubed_func, numpy_func):
    a = xp.asarray([[xp.nan, xp.nan, 1], [2, xp.nan, 3]], chunks=(2, 2), spec=spec)
    b = cubed_func(a, axis=0)
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        b.compute()
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        numpy_func(np.array([[xp.nan, xp.nan, 0], [2, xp.nan, 3]]), axis=0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("cubed_func", "numpy_func"),
    [
        (cubed.nanmax, np.nanmax),
        (cubed.nanmean, np.nanmean),
        (cubed.nanmin, np.nanmin),
        (cubed.nanstd, np.nanstd),
        (cubed.nanvar, np.nanvar),
    ],
)
def test_nan_function_allnan_warning(spec, cubed_func, numpy_func):
    a = xp.asarray([xp.nan], spec=spec)
    b = cubed_func(a)
    assert_array_equal(b.compute(), numpy_func(np.array([np.nan])))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("cubed_func", "numpy_func"),
    [
        (cubed.nancumprod, np.nancumprod),
        (cubed.nancumsum, np.nancumsum),
        (cubed.nanprod, np.nanprod),
        (cubed.nansum, np.nansum),
    ],
)
def test_nan_function_allnan_no_warning(spec, cubed_func, numpy_func):
    a = xp.asarray([xp.nan], spec=spec)
    b = cubed_func(a)
    assert_array_equal(b.compute(), numpy_func(np.array([np.nan])))


@st.composite
def nan_array(draw):
    return draw(
        xps.arrays(
            xp.float64,
            (3, 3),
            elements=st.one_of(st.floats(1, 10, width=64), st.just(np.nan)),
        )
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("cubed_func", "numpy_func"),
    [
        (cubed.nancumprod, np.nancumprod),
        (cubed.nancumsum, np.nancumsum),
        (cubed.nanmax, np.nanmax),
        (cubed.nanmean, np.nanmean),
        (cubed.nanmin, np.nanmin),
        (cubed.nanprod, np.nanprod),
        (cubed.nanstd, np.nanstd),
        (cubed.nansum, np.nansum),
        (cubed.nanvar, np.nanvar),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
@given(na=nan_array())
def test_nan_functions_hypothesis(na, cubed_func, numpy_func, axis):
    a = cubed.asarray(na, chunks=(2, 2))
    b = cubed_func(a, axis=axis)
    assert_allclose(b.compute(), numpy_func(na, axis=axis), atol=1e-08)


@pytest.mark.parametrize(
    ("cubed_func", "numpy_func"),
    [
        (cubed.nanargmax, np.nanargmax),
        (cubed.nanargmin, np.nanargmin),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
@given(na=nan_array())
def test_nanarg_functions_hypothesis(na, cubed_func, numpy_func, axis):
    a = cubed.asarray(na, chunks=(2, 2))
    b = cubed_func(a, axis=axis)

    # if numpy raises then cubed should too, otherwise the results
    # should be the same
    try:
        np_res = numpy_func(na, axis=axis)
        assert_allclose(b.compute(), np_res, atol=1e-08)
    except ValueError:
        with pytest.raises(ValueError, match="All-NaN slice encountered"):
            b.compute()
