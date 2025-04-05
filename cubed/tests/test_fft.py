import pytest
from numpy.testing import assert_allclose

import cubed
import cubed.array_api as xp
from cubed.backend_array_api import namespace as nxp


@pytest.mark.parametrize("funcname", ["fft", "ifft"])
@pytest.mark.parametrize("n", [None, 5, 13])
def test_fft(funcname, n):
    nxp_fft = getattr(nxp.fft, funcname)
    cb_fft = getattr(xp.fft, funcname)

    an = nxp.arange(100).reshape(10, 10)
    bn = nxp_fft(an, n=n)
    a = cubed.from_array(an, chunks=(1, 10))
    b = cb_fft(a, n=n)

    assert_allclose(b.compute(), bn)


def test_fft_chunked_axis_fails():
    an = nxp.arange(100).reshape(10, 10)
    a = cubed.from_array(an, chunks=(1, 10))

    with pytest.raises(ValueError):
        xp.fft.fft(a, axis=0)
