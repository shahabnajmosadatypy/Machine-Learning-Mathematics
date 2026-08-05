# 🧮 Machine Learning Mathematics — University Lecture Notes & Code

This project delivers university-lecture-note-style (**jozveh**) educational material and executable Python implementations for foundational Machine Learning Mathematics. Focusing on Linear Algebra and Spectral Decompositions, it bridges the gap between formal mathematical proofs and applied code using NumPy, SymPy, SciPy, and Matplotlib. 🚀

## 🎯 Motivation

Linear algebra is the foundational mathematical backbone of modern machine learning and artificial intelligence—powering data representations, dimensionality reduction (PCA, SVD), spectral graph theory, and optimization algorithms. This repository provides thorough, self-contained study notes and clean code implementations, allowing students and ML engineers to master linear algebra concepts intuitively and rigorously without needing external textbooks. 🧠✨

## 📂 Repository Structure

```text
Machine Learning Mathematics/
├── 📝 README.md               # Comprehensive project documentation
├── 📦 requirements.txt        # Python package dependencies
├── 🙈 .gitignore              # Git ignore rules for ML & Python artifacts
├── ⚖️ LICENSE                 # MIT License
├── 📁 data/                   # Datasets directory (synthetically generated)
│   └── 📖 README.md           # Dataset documentation
├── 📓 notebooks/              # Lecture-note (jozveh) Jupyter notebooks
│   ├── 📐 1.Vector.ipynb      # Vector spaces, inner products, and norms
│   ├── 🔢 2.Matrix.ipynb      # Matrix algebra, determinants, and inverses
│   ├── ✖️ 3.Linear_System.ipynb # Matrix equations Ax = b and linear solvers
│   ├── ⚛️ 4.1.Eigenvalues_and_Eigenvectors.ipynb # Spectral theory & eigendecomposition
│   └── 📐 4.2.SVD.ipynb       # Singular Value Decomposition & low-rank approximation
├── 💻 src/                    # Reusable Python modules
│   ├── 🟢 __init__.py         # Package initializer
│   ├── 📊 visualization.py    # Vector and linear system plot helpers
│   └── ⚡ solvers.py          # Numerical verification and SVD solvers
└── 🧪 tests/                  # Unit tests for source code helpers
    └── 🧪 test_solvers.py     # Pytest suite for solver verification
```

## 🛠️ Setup Instructions

1. **Clone this repository** to your local environment:
   ```bash
   git clone https://github.com/shahabnajmosadatypy/Machine-Learning-Mathematics.git
   cd "Machine Learning Mathematics"
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run unit tests** (optional verification):
   ```bash
   pytest
   ```

## 🏃 How to Run

Launch Jupyter Notebook and navigate through the study notebooks in sequence:
```bash
jupyter notebook
```

1. 🥇 **`notebooks/1.Vector.ipynb`**: Learn vector spaces $v \\in \\mathbb{R}^n$, linear combinations, Hadamard products, dot products $u \\cdot v$, and Euclidean norms $\\|u\\|_2$.
2. 🥈 **`notebooks/2.Matrix.ipynb`**: Explore 2D matrix representations $A \\in \\mathbb{R}^{m \\times n}$, matrix multiplication, determinants $\\det(A)$, inverses $A^{-1}$, and traces $\\text{Tr}(A)$.
3. 🥉 **`notebooks/3.Linear_System.ipynb`**: Express linear equations as $Ax = b$ and solve them symbolically with SymPy and numerically with `np.linalg.solve`.
4. 🏅 **`notebooks/4.1.Eigenvalues_and_Eigenvectors.ipynb`**: Master characteristic equations $\\det(A - \\lambda I) = 0$, invariant eigenspaces $Av = \\lambda v$, and diagonalizations.
5. 🏆 **`notebooks/4.2.SVD.ipynb`**: Understand rectangular factorizations $A = U \\Sigma V^T$, singular values $\\sigma_i = \\sqrt{\\lambda_i}$, and low-rank matrix reconstructions.

## 📊 Key Results & Empirical Verification

- ⚡ **Numerical Stability**: `np.linalg.solve(A, b)` provides exact numerical solutions (residual norm $< 10^{-15}$) for linear systems $Ax = b$, outperforming explicit matrix inversion $A^{-1}b$.
- 📉 **Geometric Alignment**: Matplotlib visual plots in `src/visualization.py` visually confirm that line/plane intersections match analytical SymPy solutions.
- 📐 **Spectral Reconstruction**: SVD matrix factorizations $A = U \\Sigma V^T$ reconstruct arbitrary $m \\times n$ real matrices with zero error norm.

## 📚 Math & Theory Overview

The project covers the core mathematical principles of machine learning:

- ➡️ **Vector Spaces & Inner Products**: Data vectors $u, v \\in \\mathbb{R}^n$ have dot product $u \\cdot v = \\|u\\|_2 \\|v\\|_2 \\cos(\\theta)$ measuring feature similarity and alignment angle.
- 🔄 **Linear Transformations**: Matrix-vector multiplication $Ax$ models spatial linear transformations (rotation, scaling, shear).
- ✖️ **Linear Systems**: Systems of linear constraints are formulated as $Ax = b$. Unique solutions exist if and only if matrix $A$ is non-singular ($\\det(A) \\neq 0$).
- ⚛️ **Eigendecomposition**: Eigenvectors $v$ satisfy $Av = \\lambda v$, defining invariant direction axes and scalar stretch factors $\\lambda$.
- 📐 **Singular Value Decomposition**: Generalizes spectral decomposition to rectangular matrices $A \\in \\mathbb{R}^{m \\times n}$ as $A = U \\Sigma V^T$, laying the foundation for Principal Component Analysis (PCA) and dimensionality reduction.

## 📜 License

This project is open-source and licensed under the terms provided in the [LICENSE](file:///c:/Users/Shahab/Documents/VSCode%20Codes/Machine%20Learning%20Mathematics/LICENSE) file. ⚖️