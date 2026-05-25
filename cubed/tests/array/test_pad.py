import numpy as np
import pytest

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal


@pytest.fixture
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


def test_pad_symmetric(spec):
    an = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = cubed.pad(a, ((1, 0), (0, 0)), mode="symmetric")
    assert b.chunks == ((2, 2), (2, 1))

    assert_array_equal(b.compute(), np.pad(an, ((1, 0), (0, 0)), mode="symmetric"))


def test_pad_constant(spec):
    an = np.arange(12).reshape(3, 4).astype(float)
    a = xp.asarray(an, chunks=(2, 2), spec=spec)

    # scalar constant_values, padding on both axes
    b = cubed.pad(a, ((2, 1), (0, 3)), mode="constant", constant_values=0)
    assert_array_equal(
        b, np.pad(an, ((2, 1), (0, 3)), mode="constant", constant_values=0)
    )

    # nan fill (xarray shift/rolling use case)
    b = cubed.pad(a, ((2, 0), (0, 0)), mode="constant", constant_values=float("nan"))
    assert_array_equal(
        b, np.pad(an, ((2, 0), (0, 0)), mode="constant", constant_values=float("nan"))
    )

    # single-axis padding (no-op on one axis)
    b = cubed.pad(a, ((0, 0), (1, 1)), mode="constant", constant_values=99)
    assert_array_equal(
        b, np.pad(an, ((0, 0), (1, 1)), mode="constant", constant_values=99)
    )
