"""
Numerical verification routines for linear systems and spectral matrix decompositions.
"""

from typing import Tuple, Dict
import numpy as np


def verify_linear_system(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """
    Verify the numerical accuracy of a linear system solution Ax = b.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (m x n).
    b : np.ndarray
        Constants vector (m,).
    x : np.ndarray
        Proposed solution vector (n,).

    Returns
    -------
    Dict[str, float]
        Dictionary containing residual vector norm and relative error.
    """
    residual = np.dot(A, x) - b
    residual_norm = float(np.linalg.norm(residual))
    b_norm = float(np.linalg.norm(b))
    relative_error = residual_norm / (b_norm + 1e-15)

    return {
        "residual_norm": residual_norm,
        "relative_error": relative_error,
        "is_valid": residual_norm < 1e-6,
    }


def compute_svd_reconstruction(
    A: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rank-k truncated SVD reconstruction of matrix A.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (m x n).
    k : int
        Target rank for truncation (1 <= k <= min(m, n)).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        U_k, S_k, Vt_k, and the reconstructed matrix A_k.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(S))

    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    A_k = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
    return U_k, S_k, Vt_k, A_k
