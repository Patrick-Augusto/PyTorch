# PyTorch Tensor Operations — Practical Exercise

**Course:** Computer Engineering — Advanced Artificial Intelligence (IA-AVAN)  
**Topic:** Deep Learning with PyTorch · Session 1: Tensor Operations

---

## Overview

This project covers the fundamentals of PyTorch tensors — the core data structure behind every deep learning model. It is structured as a Jupyter Notebook with executable code, written answers, and visualisations.

---

## Project Structure

```
TensorFlow/
├── PyTorch.ipynb               # Main notebook (submit this)
├── PyTorch..py                 # Original Python script version
├── PyTorch._Activity.pdf       # Activity specification
├── tensor_visualization.png    # Output: original vs blurred tensor heatmap
├── extension_visualization.png # Output: sine-cosine surface visualisation
├── PRD.md                      # Product Requirements Document
├── .venv/                      # Python virtual environment
└── README.md                   # This file
```

---

## Setup

### 1. Activate the virtual environment
```bash
source .venv/bin/activate
```

### 2. Install dependencies (already done if .venv exists)
```bash
pip install torch torchvision matplotlib numpy ipykernel
```

### 3. Run the notebook

**Option A — VS Code (recommended)**
- Open `PyTorch.ipynb`
- Select kernel **"Python (.venv IA-AVAN)"** (top-right corner)
- Click **Run All**

**Option B — JupyterLab**
```bash
jupyter lab
```

---

## Contents

### Part 1 — Creating and Manipulating Tensors
Creating tensors from lists and NumPy arrays, using `torch.zeros`, `torch.ones`, `torch.rand`, `torch.arange`, reshaping, transposing, and flattening.

### Part 2 — Basic Tensor Operations
Element-wise arithmetic, matrix multiplication (`torch.matmul`, `@`), dot product, and statistical operations (sum, mean, std, max, min) along specific dimensions.

### Part 3 — Tensor Operations for Machine Learning
Broadcasting across different shapes, advanced indexing/slicing, and device management (CPU / CUDA GPU).

### Part 4 — Hands-on Exercises

| #   | Exercise                               | Key function                      |
| --- | -------------------------------------- | --------------------------------- |
| 1   | 3×3 Identity matrix                    | `torch.eye(3)`                    |
| 2   | Matrix multiplication (2×3) · (3×2)    | `torch.matmul()`                  |
| 3   | Cosine similarity between two vectors  | `F.cosine_similarity()`           |
| 4   | Row normalisation (each row sums to 1) | `tensor.sum(dim=1, keepdim=True)` |
| 5   | Euclidean distance function            | `torch.sqrt(torch.sum(...))`      |

### Part 5 — Visualisation
Heatmap of a 2-D tensor before and after a 3×3 averaging convolution (blur).

### Conceptual Questions
Written answers to the five theory questions from the activity specification (Q1–Q5).

### Extension Activities
1. **Linear Regression** — pure tensor gradient descent to fit $y = 3x + 2$
2. **SVD & Eigenvalues** — `torch.linalg.svd` and `torch.linalg.eig` on a 2×2 matrix
3. **Custom Visualisation** — sine-cosine surface $f(x,y) = \sin(x)\cos(y)$

---

## Dependencies

| Package     | Version |
| ----------- | ------- |
| Python      | 3.9+    |
| torch       | 2.8.0   |
| torchvision | 0.23.0  |
| numpy       | 2.0.2   |
| matplotlib  | 3.9.4   |
| ipykernel   | latest  |

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/torch.html)
- [PyTorch Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
