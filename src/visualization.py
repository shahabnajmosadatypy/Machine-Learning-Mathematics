"""
Visualization utilities for vector arithmetic, linear systems, and function limits.
"""

from typing import List, Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt


def plot_vectors_2d(
    vectors: List[np.ndarray],
    origins: Optional[List[np.ndarray]] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: str = "2D Vector Visualization",
    xlim: Tuple[float, float] = (-5, 5),
    ylim: Tuple[float, float] = (-5, 5),
) -> plt.Figure:
    """
    Plot 2D vectors starting from specified origin points.

    Parameters
    ----------
    vectors : List[np.ndarray]
        List of 2D vectors [x, y].
    origins : Optional[List[np.ndarray]]
        List of origin points [x0, y0] for each vector. Defaults to (0,0).
    colors : Optional[List[str]]
        Color strings for each vector.
    labels : Optional[List[str]]
        Labels for the legend.
    title : str
        Title of the plot.
    xlim : Tuple[float, float]
        X-axis limits.
    ylim : Tuple[float, float]
        Y-axis limits.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    if origins is None:
        origins = [np.array([0, 0]) for _ in vectors]

    if colors is None:
        colors = ["blue", "red", "green", "purple", "orange"][: len(vectors)]

    for i, (vec, orig) in enumerate(zip(vectors, origins)):
        lbl = labels[i] if labels and i < len(labels) else f"v_{i+1}"
        ax.quiver(
            orig[0],
            orig[1],
            vec[0],
            vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colors[i % len(colors)],
            label=f"{lbl} = {vec}",
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return fig


def plot_linear_system_2d(
    A: np.ndarray,
    b: np.ndarray,
    solution: Optional[np.ndarray] = None,
    x_range: Tuple[float, float] = (-5, 5),
    title: str = "2D Linear System Visualization",
) -> plt.Figure:
    """
    Plot lines corresponding to a 2D linear system Ax = b.

    Parameters
    ----------
    A : np.ndarray
        2x2 coefficient matrix.
    b : np.ndarray
        2x1 constants vector.
    solution : Optional[np.ndarray]
        Computed solution point [x1, x2].
    x_range : Tuple[float, float]
        Range of x1 values for plotting.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    x1 = np.linspace(x_range[0], x_range[1], 400)

    # Line 1: a11*x1 + a12*x2 = b1  =>  x2 = (b1 - a11*x1) / a12
    if A[0, 1] != 0:
        x2_line1 = (b[0] - A[0, 0] * x1) / A[0, 1]
        ax.plot(x1, x2_line1, label=f"{A[0,0]}x₁ + {A[0,1]}x₂ = {b[0]}", color="blue")
    else:
        ax.axvline(x=b[0] / A[0, 0], label=f"{A[0,0]}x₁ = {b[0]}", color="blue")

    # Line 2: a21*x1 + a22*x2 = b2  =>  x2 = (b2 - a21*x1) / a22
    if A[1, 1] != 0:
        x2_line2 = (b[1] - A[1, 0] * x1) / A[1, 1]
        ax.plot(x1, x2_line2, label=f"{A[1,0]}x₁ + {A[1,1]}x₂ = {b[1]}", color="red")
    else:
        ax.axvline(x=b[1] / A[1, 0], label=f"{A[1,0]}x₁ = {b[1]}", color="red")

    if solution is not None:
        ax.plot(
            solution[0],
            solution[1],
            "go",
            markersize=10,
            label=f"Intersection ({solution[0]:.2f}, {solution[1]:.2f})",
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_function_limit(
    func: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (0.0, 2.0),
    target_x: Optional[float] = 1.0,
    limit_val: Optional[float] = 2.0,
    title: str = "Function Limit Visualization",
) -> plt.Figure:
    """
    Plot a function f(x) and highlight target point x -> c and limit value L.

    Parameters
    ----------
    func : Callable[[np.ndarray], np.ndarray]
        Vectorized Python function f(x).
    x_range : Tuple[float, float]
        Plotting interval for x.
    target_x : Optional[float]
        Point c where limit is evaluated.
    limit_val : Optional[float]
        Calculated limit L = lim_{x -> c} f(x).
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(x_range[0], x_range[1], 500)
    # Avoid exact division by zero at target point if present
    if target_x is not None:
        x = x[np.abs(x - target_x) > 1e-7]

    y = func(x)
    ax.plot(x, y, color="blue", linewidth=2, label="$f(x)$")

    if target_x is not None:
        ax.axvline(x=target_x, color="red", linestyle="--", alpha=0.7, label=f"$x \\to {target_x}$")
    if limit_val is not None:
        ax.axhline(y=limit_val, color="green", linestyle="--", alpha=0.7, label=f"$L = {limit_val}$")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(title)
    ax.legend()
    return fig
