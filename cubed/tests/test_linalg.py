import numpy as np
from numpy.testing import assert_allclose

import cubed
import cubed.array_api as xp


def test_qr():
    A = np.reshape(np.arange(32, dtype=np.float64), (16, 2))
    Q, R = xp.linalg.qr(xp.asarray(A, chunks=(4, 2)))

    cubed.visualize(Q, R, optimize_graph=False)
    Q, R = cubed.compute(Q, R)

    assert_allclose(Q @ R, A, atol=1e-08)
    assert_allclose(Q.T @ Q, np.eye(2, 2), atol=1e-08)  # Q must be orthonormal
    assert_allclose(R, np.triu(R), atol=1e-08)  # R must be upper triangular


def test_qr_recursion():
    spec = cubed.Spec(allowed_mem=128 * 4 * 1.5, reserved_mem=0)
    A = np.reshape(np.arange(64, dtype=np.float64), (32, 2))
    Q, R = xp.linalg.qr(xp.asarray(A, chunks=(8, 2), spec=spec))

    cubed.visualize(Q, R, optimize_graph=False)
    Q, R = cubed.compute(Q, R)

    assert_allclose(Q @ R, A, atol=1e-08)
    assert_allclose(Q.T @ Q, np.eye(2, 2), atol=1e-08)  # Q must be orthonormal
    assert_allclose(R, np.triu(R), atol=1e-08)  # R must be upper triangular
