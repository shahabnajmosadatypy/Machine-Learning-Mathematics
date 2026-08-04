# 🧮 Machine Learning Mathematics

This project demonstrates the core mathematical foundations required for Machine Learning, specifically focusing on linear algebra. It includes practical, executable implementations of vectors, matrices, and linear systems using Python, NumPy, and SymPy. 🚀

## 🎯 Motivation
Understanding linear algebra is crucial for modern machine learning, as it powers data representations, model weights, and optimization algorithms. This project provides hands-on, executable examples to bridge the gap between abstract mathematical theory and applied code, helping students and developers build a strong intuition for these concepts. 🧠✨

## 📂 Repository Structure
```text
project-root/
├── 📝 README.md               # Project documentation
├── 📦 requirements.txt        # Python dependencies
├── 🙈 .gitignore              # Ignored files and folders
├── ⚖️ LICENSE                 # Project license
├── 📓 notebooks/              # Jupyter notebooks containing math/ML code
│   ├── 📐 1.Vector.ipynb      # Vector operations and theory
│   ├── 🔢 2.Matrix.ipynb      # Matrix arithmetic, properties, and inverses
│   └── ✖️ 3.Linear_System.ipynb # Solving systems of linear equations
├── 💻 src/                    # Reusable Python modules (empty for now)
├── 📖 docs/                   # Markdown documentation (empty for now)
└── 🧪 tests/                  # Project tests (empty for now)
```

## 🛠️ Setup Instructions

1. **Clone this repository** to your local machine. 🖥️
2. **Create a virtual environment** (recommended): 🛡️
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install the dependencies**: ⚙️
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 How to Run
Launch Jupyter Notebook and explore the notebooks in the following order:
```bash
jupyter notebook
```
1. 🥇 Start with `notebooks/1.Vector.ipynb` to understand 1D structures (vectors) and basic scalar operations.
2. 🥈 Proceed to `notebooks/2.Matrix.ipynb` to explore 2D structures, matrix multiplication, and key properties (Trace, Determinant, Inverse, Norm).
3. 🥉 Finish with `notebooks/3.Linear_System.ipynb` to see how vectors and matrices are combined to computationally solve multi-variable systems of equations.

## 📊 Key Results / Findings
- ⚡ The notebooks successfully demonstrate that numerical solvers (`np.linalg.solve` and `scipy.linalg.solve`) can efficiently compute the intersections of lines and planes, seamlessly replacing manual algebraic substitutions.
- 📉 Visualizing a 2D linear system using `matplotlib` geometrically confirms the exact analytic solutions computed by `sympy`.

## 📚 Math/Theory Overview
The project heavily utilizes the following core linear algebra concepts:
- ➡️ **Vectors & Matrices**: Data is mathematically represented as vectors $v \in \mathbb{R}^n$ and matrices $A \in \mathbb{R}^{m \times n}$.
- 🔄 **Matrix Multiplication**: Defines mathematical transformations and projections, computed using the dot product as $C_{i,j} = \sum_k A_{i,k}B_{k,j}$.
- ✖️ **Linear Systems**: Real-world problems are formulated as the matrix equation $Ax = b$. If $A$ is an invertible square matrix, the solution is exactly $x = A^{-1}b$.
- 🧮 **Determinants & Inverses**: A square matrix system has a unique solution if and only if $\det(A) \neq 0$.

## 📜 License
This project is licensed under the terms provided in the `LICENSE` file. ⚖️