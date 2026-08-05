"""
Unit tests for source code solver routines.
"""

import numpy as np
from src.solvers import verify_linear_system, compute_svd_reconstruction


def test_verify_linear_system():
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 5], dtype=float)
    x = np.linalg.solve(A, b)

    result = verify_linear_system(A, b, x)
    assert result["is_valid"]
    assert result["residual_norm"] < 1e-10


def test_compute_svd_reconstruction():
    A = np.array([[3, 1, 1], [-1, 3, 1]], dtype=float)
    U_k, S_k, Vt_k, A_reconstructed = compute_svd_reconstruction(A, k=2)

    assert np.allclose(A, A_reconstructed)
    assert U_k.shape == (2, 2)
    assert len(S_k) == 2
