import numpy as np
from numpy.testing import assert_allclose

import cubed
import cubed.array_api as xp
from cubed.array_api import linalg  # noqa: F401


def test_qr():
    A = np.reshape(np.arange(32, dtype=np.float64), (16, 2))
    Q, R = xp.linalg.qr(xp.asarray(A, chunks=(4, 2)))

    cubed.visualize(Q, R, optimize_graph=False)
    Q, R = cubed.compute(Q, R)

    assert_allclose(Q @ R, A)
    assert_allclose(Q.T @ Q, np.eye(2, 2), atol=1e-08)  # Q must be orthonormal
    assert_allclose(R, np.triu(R), atol=1e-08)  # R must be upper triangular
