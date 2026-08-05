"""
Machine Learning Mathematics Source Package
Provides reusable helper routines for vector visualization, matrix transformations,
linear system solvers, and spectral decompositions.
"""

from .visualization import plot_vectors_2d, plot_linear_system_2d
from .solvers import verify_linear_system, compute_svd_reconstruction

__all__ = [
    "plot_vectors_2d",
    "plot_linear_system_2d",
    "verify_linear_system",
    "compute_svd_reconstruction",
]
