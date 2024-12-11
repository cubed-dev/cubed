import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed.backend_array_api import namespace as nxp


@pytest.mark.parametrize("n", [None, 5, 13])
def test_fft(n):
    an = np.arange(100).reshape(10, 10)
    bn = nxp.fft.fft(an, n=n)
    a = cubed.from_array(an, chunks=(1, 10))
    b = xp.fft.fft(a, n=n)

    assert_array_equal(b.compute(), bn)


def test_fft_chunked_axis_fails():
    an = np.arange(100).reshape(10, 10)
    a = cubed.from_array(an, chunks=(1, 10))

    with pytest.raises(ValueError):
        xp.fft.fft(a, axis=0)
