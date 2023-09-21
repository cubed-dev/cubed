import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


def test_pad(spec):
    an = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = cubed.pad(a, ((1, 0), (0, 0)), mode="symmetric")
    assert b.chunks == ((2, 2), (2, 1))

    assert_array_equal(b.compute(), np.pad(an, ((1, 0), (0, 0)), mode="symmetric"))
