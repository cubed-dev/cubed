import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


def test_nanmean(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, xp.nan]], chunks=(2, 2), spec=spec)
    b = cubed.nanmean(a)
    assert_array_equal(
        b.compute(), np.nanmean(np.array([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]]))
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nanmean_allnan(spec):
    a = xp.asarray([xp.nan], spec=spec)
    b = cubed.nanmean(a)
    assert_array_equal(b.compute(), np.nanmean(np.array([np.nan])))


def test_nansum(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, xp.nan]], chunks=(2, 2), spec=spec)
    b = cubed.nansum(a)
    assert_array_equal(
        b.compute(), np.nansum(np.array([[1, 2, 3], [4, 5, 6], [7, 8, np.nan]]))
    )


def test_nansum_allnan(spec):
    a = xp.asarray([xp.nan], spec=spec)
    b = cubed.nansum(a)
    assert_array_equal(b.compute(), np.nansum(np.array([np.nan])))
