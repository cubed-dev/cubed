import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed.array_api as xp
from cubed.testing import raise_if_computes


def test_raise_if_computes():
    # shouldn't raise since compute has not been called
    with raise_if_computes():
        a = xp.ones((3, 3), chunks=(2, 2))
        b = xp.negative(a)

    # should raise since compute is called
    with pytest.raises(RuntimeError):
        with raise_if_computes():
            b.compute()

    # shouldn't raise since we are outside the context manager
    assert_array_equal(b.compute(), -np.ones((3, 3)))
