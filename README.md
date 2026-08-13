# 🧮 Machine Learning Mathematics — University Lecture Notes & Code

This project delivers comprehensive university lecture notes and executable Python implementations for foundational Machine Learning Mathematics. Focusing on Linear Algebra, Calculus, Integration, and Optimization, it bridges the gap between formal mathematical proofs and applied code using NumPy, SymPy, SciPy, and Matplotlib. 🚀

## 🎯 Motivation

Mathematics is the foundational backbone of modern machine learning and artificial intelligence—powering data representations, optimization algorithms, loss functions, dimensionality reduction (PCA, SVD), and spectral graph theory. This repository provides thorough, self-contained study notes and clean code implementations, allowing students and ML engineers to master core mathematical concepts intuitively and rigorously without needing external textbooks. 🧠✨

## 📂 Repository Structure

```text
Machine Learning Mathematics/
├── 📝 README.md               # Comprehensive project documentation
├── 📦 requirements.txt        # Python package dependencies
├── 🙈 .gitignore              # Git ignore rules for ML & Python artifacts
├── ⚖️ LICENSE                 # MIT License
├── 📁 data/                   # Datasets directory (optional local data storage)
├── 📓 notebooks/              # University lecture note Jupyter notebooks
│   ├── 📐 1.Vector.ipynb      # Vector spaces, inner products, and norms
│   ├── 🔢 2.Matrix.ipynb      # Matrix algebra, determinants, and inverses
│   ├── ✖️ 3.Linear_System.ipynb # Matrix equations Ax = b and linear solvers
│   ├── ⚛️ 4.1.Eigenvalues_and_Eigenvectors.ipynb # Spectral theory & eigendecomposition
│   ├── 📐 4.2.SVD.ipynb       # Singular Value Decomposition & low-rank approximation
│   ├── 📈 5.Limit.ipynb       # Limits of functions, removable discontinuities, & asymptotes
│   ├── ⚡ 6.Derivative.ipynb  # Ordinary & partial derivatives, Chain Rule, & log/exp gradients
│   ├──  7.Integrate.ipynb   # Indefinite/definite integration & area computation
│   ├── 📉 8.1.Gradient_Descent.ipynb # Gradient descent update rules & learning rates
│   ├── 🎯 8.2.Optimization.ipynb     # Constrained optimization & Lagrange multipliers
│   ├── 📊 9.1.Statistics.ipynb       # Descriptive statistics, dispersion, covariance, & Pearson r
│   └── 🎲 9.2.Distribution.ipynb     # Probability distributions: Bernoulli, Binomial, Poisson, Normal
├── 💻 src/                    # Reusable Python modules
│   ├── 🟢 __init__.py         # Package initializer
│   └── 🛠️ utils.py            # Consolidated numerical solvers and plotting utilities
└── 🧪 tests/                  # Unit tests for source code helpers
    └── 🧪 test_utils.py       # Pytest / unittest suite for utils verification
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
   python -m unittest tests/test_utils.py
   ```

## 🏃 How to Run

Launch Jupyter Notebook and navigate through the study notebooks in sequence:
```bash
jupyter notebook
```

1. 🥇 **`notebooks/1.Vector.ipynb`**: Learn vector spaces $v \in \mathbb{R}^n$, linear combinations, Hadamard products, dot products $u \cdot v$, and Euclidean norms $\|u\|_2$.
2. 🥈 **`notebooks/2.Matrix.ipynb`**: Explore 2D matrix representations $A \in \mathbb{R}^{m \times n}$, matrix multiplication, determinants $\det(A)$, inverses $A^{-1}$, and traces $\text{Tr}(A)$.
3. 🥉 **`notebooks/3.Linear_System.ipynb`**: Express linear equations as $Ax = b$ and solve them symbolically with SymPy and numerically with `np.linalg.solve`.
4. 🏅 **`notebooks/4.1.Eigenvalues_and_Eigenvectors.ipynb`**: Master characteristic equations $\det(A - \lambda I) = 0$, invariant eigenspaces $Av = \lambda v$, and diagonalizations.
5. 🏆 **`notebooks/4.2.SVD.ipynb`**: Understand rectangular factorizations $A = U \Sigma V^T$, singular values $\sigma_i = \sqrt{\lambda_i}$, and low-rank matrix reconstructions.
6. 🎯 **`notebooks/5.Limit.ipynb`**: Study functional limits $\lim_{x \to c} f(x) = L$, removable discontinuities ($\frac{0}{0}$ indeterminate forms), and asymptotic limits at infinity ($\lim_{x \to \infty} \frac{1}{x} = 0$).
7. ⚡ **`notebooks/6.Derivative.ipynb`**: Master ordinary $f'(x)$ and partial $\frac{\partial f}{\partial x}$ derivatives, Chain Rule backpropagation, and logarithmic/exponential loss gradients.
8.  **`notebooks/7.Integrate.ipynb`**: Study indefinite and definite integrals $\int_{a}^{b} f(x) \, dx = F(b) - F(a)$, partial integration, and area under curves.
9. 📉 **`notebooks/8.1.Gradient_Descent.ipynb`**: Master the parameter update rule $x^{(k+1)} = x^{(k)} - \eta \nabla f(x^{(k)})$, learning rate dynamics, and multivariate loss optimization.
10. 🎯 **`notebooks/8.2.Optimization.ipynb`**: Study constrained minimization $\min_x f(x) \text{ s.t. } g(x) = 0$, SciPy SLSQP optimization, and the Method of Lagrange Multipliers $\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda g(x, y)$.
11. 📊 **`notebooks/9.1.Statistics.ipynb`**: Study central tendency (mean, median, mode), IQR outlier bounds, variance/standard deviation, covariance, and Pearson correlation $r$.
12. 🎲 **`notebooks/9.2.Distribution.ipynb`**: Explore discrete (Bernoulli, Binomial, Poisson) and continuous (Normal Gaussian) distributions with empirical PDF fitting.

## 📊 Key Results & Empirical Verification

- ⚡ **Numerical Stability**: `np.linalg.solve(A, b)` provides exact numerical solutions (residual norm $< 10^{-15}$) for linear systems $Ax = b$, outperforming explicit matrix inversion $A^{-1}b$.
- 📉 **Geometric Alignment**: Matplotlib visual plots in `src/utils.py` visually confirm line/plane intersections, function limits, definite integral areas, and probability distributions.
- 📐 **Spectral Reconstruction**: SVD matrix factorizations $A = U \Sigma V^T$ reconstruct arbitrary $m \times n$ real matrices with zero error norm.

## 📚 Math & Theory Overview

The project covers the core mathematical principles of machine learning:

- ➡️ **Vector Spaces & Inner Products**: Data vectors $u, v \in \mathbb{R}^n$ have dot product $u \cdot v = \|u\|_2 \|v\|_2 \cos(\theta)$ measuring feature similarity and alignment angle.
- 🔄 **Linear Transformations**: Matrix-vector multiplication $Ax$ models spatial linear transformations (rotation, scaling, shear).
- ✖️ **Linear Systems**: Systems of linear constraints are formulated as $Ax = b$. Unique solutions exist if and only if matrix $A$ is non-singular ($\det(A) \neq 0$).
- ⚛️ **Eigendecomposition**: Eigenvectors $v$ satisfy $Av = \lambda v$, defining invariant direction axes and scalar stretch factors $\lambda$.
- 📐 **Singular Value Decomposition**: Generalizes spectral decomposition to rectangular matrices $A \in \mathbb{R}^{m \times n}$ as $A = U \Sigma V^T$, laying the foundation for Principal Component Analysis (PCA) and dimensionality reduction.
- 📈 **Limits & Calculus Foundations**: Limits $\lim_{x \to c} f(x) = L$ quantify continuous convergence, resolving indeterminate forms $\frac{0}{0}$.
- ⚡ **Derivatives & Optimization**: Ordinary and partial derivatives $\frac{\partial f}{\partial x_i}$ construct the gradient vector $\nabla f$, while the Chain Rule $\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)$ powers neural network backpropagation.
-  **Integration & Cumulative Distribution**: Integrals $\int f(x) \, dx$ compute probability density areas and expected values.
- 📉 **Gradient Descent & Constrained Optimization**: Iterative update rules $x^{(k+1)} = x^{(k)} - \eta \nabla f(x^{(k)})$ minimize loss functions, while Lagrange Multipliers $\mathcal{L}(x, y, \lambda)$ handle constrained weight surfaces.
- 📊 **Descriptive Statistics & Covariance**: Measures of central tendency, IQR outlier filtering, and Pearson correlation coefficient $r \in [-1, 1]$ measure feature dependencies.
- 🎲 **Probability Distributions**: Discrete PMFs (Bernoulli, Binomial, Poisson) and continuous Gaussian PDFs model random data generation and noise in ML pipelines.

## 📜 License

This project is open-source and licensed under the terms provided in the [LICENSE](file:///c:/Users/Shahab/Documents/VSCode%20Codes/Machine%20Learning%20Mathematics/LICENSE) file. ⚖️