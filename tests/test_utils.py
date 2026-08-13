"""
Unit tests for consolidated utilities in src/utils.py.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.utils import (
    verify_linear_system,
    compute_svd_reconstruction,
    plot_vectors_2d,
    plot_linear_system_2d,
    plot_function_limit,
    plot_area_under_curve,
)


class TestUtils(unittest.TestCase):
    def test_verify_linear_system(self):
        A = np.array([[1.0, -1.0], [3.0, 1.0]])
        b = np.array([1.0, 9.0])
        x = np.array([2.5, 1.5])
        res = verify_linear_system(A, b, x)
        self.assertTrue(res["is_valid"])
        self.assertLess(res["residual_norm"], 1e-10)

    def test_compute_svd_reconstruction(self):
        A = np.array([[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0]])
        U_k, S_k, Vt_k, A_k = compute_svd_reconstruction(A, k=2)
        self.assertEqual(A_k.shape, A.shape)
        np.testing.assert_allclose(A, A_k, atol=1e-10)

    def test_plot_vectors_2d(self):
        v1 = np.array([3, 4])
        v2 = np.array([7, 2])
        fig = plot_vectors_2d([v1, v2], labels=["u", "v"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_linear_system_2d(self):
        A = np.array([[1.0, -1.0], [3.0, 1.0]])
        b = np.array([1.0, 9.0])
        sol = np.array([2.5, 1.5])
        fig = plot_linear_system_2d(A, b, solution=sol)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_function_limit(self):
        func = lambda x: (x**2 - 1) / (x - 1)
        fig = plot_function_limit(func, x_range=(0.0, 2.0), target_x=1.0, limit_val=2.0)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_area_under_curve(self):
        func = lambda x: x
        fig = plot_area_under_curve(func, x_range=(0.0, 3.0), integration_bounds=(1.0, 2.0))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
