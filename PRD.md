# PRD — PyTorch Tensor Operations Activity

**Course:** Computer Engineering — Neural Networks (IA-AVAN)  
**File:** `activity1.py`  
**Date:** 2026-05-05

---

## 1. Objective

Complete the five hands-on exercises in `activity1.py` (Part 4) that currently contain `TODO` placeholders, and ensure the full script runs without errors, including the visualization in Part 5.

---

## 2. Scope

| #   | Exercise                                              | Status |
| --- | ----------------------------------------------------- | ------ |
| 1   | Create a 3×3 identity matrix                          | TODO   |
| 2   | Matrix multiplication (2×3) × (3×2)                   | TODO   |
| 3   | Cosine similarity between two vectors                 | TODO   |
| 4   | Normalize matrix rows (each row sums to 1)            | TODO   |
| 5   | Euclidean distance function between two tensor points | TODO   |

Out of scope: changes to Parts 1–3 and Part 5 (already implemented).

---

## 3. Functional Requirements

### Exercise 1 — Identity Matrix
- Use `torch.eye(3)` to create a 3×3 identity matrix.
- Print the result with a descriptive label.

### Exercise 2 — Matrix Multiplication
- Create `matrix_a` of shape (2, 3) with `torch.rand` or explicit values.
- Create `matrix_b` of shape (3, 2) with `torch.rand` or explicit values.
- Compute `result = torch.matmul(matrix_a, matrix_b)` → shape (2, 2).
- Print both matrices and the result.

### Exercise 3 — Cosine Similarity
- Create two 1-D float tensors (`vector_1`, `vector_2`) of the same length.
- Compute cosine similarity using `torch.nn.functional.cosine_similarity` with `dim=0`.
- Print the similarity value.

### Exercise 4 — Row Normalization
- Create a 3×4 float matrix (random or explicit).
- Divide each row by its sum so every row sums to 1.  
  Formula: `normalized = matrix / matrix.sum(dim=1, keepdim=True)`
- Print the normalized matrix and verify row sums.

### Exercise 5 — Euclidean Distance Function
- Define `euclidean_distance(point1, point2)` that returns a scalar tensor.  
  Formula: $d = \sqrt{\sum_i (p1_i - p2_i)^2}$ → `torch.sqrt(torch.sum((point1 - point2) ** 2))`
- Test the function with two sample 3-D points and print the result.

---

## 4. Non-Functional Requirements

- No new dependencies beyond those already imported (`torch`, `numpy`, `matplotlib`).
- All exercises must print clear, labelled output.
- Code must be compatible with PyTorch ≥ 1.8.

---

## 5. Acceptance Criteria

- [ ] Running `python activity1.py` completes with exit code 0.
- [ ] Each exercise prints a non-empty, correct result.
- [ ] `tensor_visualization.png` is saved successfully.
- [ ] No `TODO` comments remain in Part 4.
