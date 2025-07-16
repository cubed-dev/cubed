import numpy as np
import pytest
from numpy.testing import assert_allclose

import cubed
import cubed.array_api as xp
from cubed.core.plan import arrays_to_plan


def test_qr():
    A = np.reshape(np.arange(32, dtype=np.float64), (16, 2))
    Q, R = xp.linalg.qr(xp.asarray(A, chunks=(4, 2)))

    plan_unopt = arrays_to_plan(Q, R)._finalize()
    assert plan_unopt.num_primitive_ops() == 4

    Q, R = cubed.compute(Q, R)

    assert_allclose(Q @ R, A, atol=1e-08)
    assert_allclose(Q.T @ Q, np.eye(2, 2), atol=1e-08)  # Q must be orthonormal
    assert_allclose(R, np.triu(R), atol=1e-08)  # R must be upper triangular


def test_qr_recursion():
    A = np.reshape(np.arange(128, dtype=np.float64), (64, 2))

    # find a memory setting where recursion happens
    found = False
    for factor in range(4, 16):
        spec = cubed.Spec(allowed_mem=128 * factor, reserved_mem=0)

        try:
            Q, R = xp.linalg.qr(xp.asarray(A, chunks=(8, 2), spec=spec))

            found = True
            plan_unopt = arrays_to_plan(Q, R)._finalize()
            assert plan_unopt.num_primitive_ops() > 4  # more than without recursion

            Q, R = cubed.compute(Q, R)

            assert_allclose(Q @ R, A, atol=1e-08)
            assert_allclose(Q.T @ Q, np.eye(2, 2), atol=1e-08)  # Q must be orthonormal
            assert_allclose(R, np.triu(R), atol=1e-08)  # R must be upper triangular

            break

        except ValueError:
            pass  # not enough memory

    assert found


def test_qr_chunking():
    A = xp.ones((32, 4), chunks=(4, 2))
    with pytest.raises(
        ValueError,
        match=r"qr only supports tall-and-skinny \(single column chunk\) arrays.",
    ):
        xp.linalg.qr(A)


def test_svd():
    A = np.reshape(np.arange(32, dtype=np.float64), (16, 2))

    U, S, Vh = xp.linalg.svd(xp.asarray(A, chunks=(4, 2)), full_matrices=False)
    U, S, Vh = cubed.compute(U, S, Vh)

    assert_allclose(U * S @ Vh, A, atol=1e-08)
    assert_allclose(U.T @ U, np.eye(2, 2), atol=1e-08)  # U must be orthonormal
    assert_allclose(Vh @ Vh.T, np.eye(2, 2), atol=1e-08)  # Vh must be orthonormal


def test_svd_recursion():
    A = np.reshape(np.arange(128, dtype=np.float64), (64, 2))

    # find a memory setting where recursion happens
    found = False
    for factor in range(4, 16):
        spec = cubed.Spec(allowed_mem=128 * factor, reserved_mem=0)

        try:
            U, S, Vh = xp.linalg.svd(
                xp.asarray(A, chunks=(8, 2), spec=spec), full_matrices=False
            )

            found = True
            plan_unopt = arrays_to_plan(U, S, Vh)._finalize()
            assert plan_unopt.num_primitive_ops() > 4  # more than without recursion

            U, S, Vh = cubed.compute(U, S, Vh)

            assert_allclose(U * S @ Vh, A, atol=1e-08)
            assert_allclose(U.T @ U, np.eye(2, 2), atol=1e-08)  # U must be orthonormal
            assert_allclose(
                Vh @ Vh.T, np.eye(2, 2), atol=1e-08
            )  # Vh must be orthonormal

            break

        except ValueError:
            pass  # not enough memory

    assert found
