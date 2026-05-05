import torch
import matplotlib.pyplot as plt
import numpy as np

print("PART 1: CREATING AND MANIPULATING TENSORS")
print("-----------------------------------------")

print("1.1 Creating tensors from different data sources:")
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D tensor: {tensor_1d}")
print(f"Shape: {tensor_1d.shape}")

tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D tensor:\n{tensor_2d}")
print(f"Shape: {tensor_2d.shape}")

numpy_array = np.array([[1.5, 2.5], [3.5, 4.5]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy:\n{tensor_from_numpy}")
print(f"Data type: {tensor_from_numpy.dtype}")

print("\n1.2 Tensor creation functions:")
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_tensor = torch.rand(2, 3)

print(f"Zeros tensor:\n{zeros}")
print(f"Ones tensor:\n{ones}")
print(f"Random tensor:\n{random_tensor}")

range_tensor = torch.arange(0, 10, step=2)
print(f"Range tensor: {range_tensor}")

float_tensor = torch.ones(2, 2, dtype=torch.float32)
int_tensor = torch.ones(2, 2, dtype=torch.int32)
print(f"Float tensor:\n{float_tensor}")
print(f"Integer tensor:\n{int_tensor}")

print("\n1.3 Tensor reshaping:")
original = torch.arange(12)
print(f"Original tensor: {original}, shape: {original.shape}")

reshaped = original.reshape(3, 4)
print(f"Reshaped to 3x4:\n{reshaped}")

viewed = original.view(4, 3)
print(f"Viewed as 4x3:\n{viewed}")

transposed = reshaped.T
print(f"Transposed:\n{transposed}")

flattened = transposed.flatten()
print(f"Flattened: {flattened}")

print("\nPART 2: BASIC TENSOR OPERATIONS")
print("------------------------------")

print("2.1 Element-wise operations:")
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"Tensor a:\n{a}")
print(f"Tensor b:\n{b}")

print(f"a + b:\n{a + b}")
print(f"torch.add(a, b):\n{torch.add(a, b)}")
print(f"a - b:\n{a - b}")
print(f"a * b:\n{a * b}")
print(f"a / b:\n{a / b}")
print(f"a^2:\n{a ** 2}")

print("\n2.2 Matrix operations:")
mat_mul = torch.matmul(a, b)
print(f"Matrix multiplication (a @ b):\n{mat_mul}")
print(f"Same using torch.matmul(a, b):\n{torch.matmul(a, b)}")

c = torch.tensor([1, 2, 3])
d = torch.tensor([4, 5, 6])
print(f"Dot product of {c} and {d}: {torch.dot(c, d)}")

print("\n2.3 Statistical operations:")
sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print(f"Sample tensor:\n{sample_tensor}")

print(f"Sum of all elements: {sample_tensor.sum()}")
print(f"Mean value: {sample_tensor.mean()}")
print(f"Standard deviation: {sample_tensor.std()}")
print(f"Maximum value: {sample_tensor.max()}")
print(f"Minimum value: {sample_tensor.min()}")
print(f"Sum along rows (dim=0):\n{sample_tensor.sum(dim=0)}")
print(f"Sum along columns (dim=1):\n{sample_tensor.sum(dim=1)}")

print("\nPART 3: TENSOR OPERATIONS FOR MACHINE LEARNING")
print("--------------------------------------------")

print("3.1 Broadcasting:")
matrix = torch.rand(3, 4)
vector = torch.rand(4)

print(f"Matrix shape: {matrix.shape}")
print(f"Vector shape: {vector.shape}")

result = matrix + vector
print(f"Result shape after broadcasting: {result.shape}")
print(f"First few values:\n{result[:2, :2]}")

print("\n3.2 Indexing and slicing:")
data = torch.arange(16).reshape(4, 4)
print(f"Original data:\n{data}")

print(f"Element at position [1, 2]: {data[1, 2]}")
print(f"First row: {data[0]}")
print(f"Last column: {data[:, -1]}")
print(f"Top-left 2x2 block:\n{data[:2, :2]}")
print(f"Bottom-right 2x2 block:\n{data[2:, 2:]}")

indices = torch.tensor([0, 2])
print(f"Rows 0 and 2:\n{data[indices]}")

print("\n3.3 Device management:")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

x = torch.rand(3, 3, device=device)
print(f"Tensor created on {device}:\n{x}")

y = torch.rand(3, 3)
y_device = y.to(device)
print(f"Tensor moved to {device}")

print("\nPART 4: EXERCISES")
print("---------------")

print("\nExercise 1: Create a 3x3 identity matrix")
identity_matrix = torch.eye(3)
print(f"Identity matrix:\n{identity_matrix}")

print("\nExercise 2: Matrix multiplication")
matrix_a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
matrix_b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float32)
result = torch.matmul(matrix_a, matrix_b)
print(f"matrix_a (2x3):\n{matrix_a}")
print(f"matrix_b (3x2):\n{matrix_b}")
print(f"Result (2x2):\n{result}")

print("\nExercise 3: Cosine similarity")
vector_1 = torch.tensor([1.0, 2.0, 3.0])
vector_2 = torch.tensor([4.0, 5.0, 6.0])
cosine_sim = torch.nn.functional.cosine_similarity(vector_1.unsqueeze(0), vector_2.unsqueeze(0))
print(f"vector_1: {vector_1}")
print(f"vector_2: {vector_2}")
print(f"Cosine similarity: {cosine_sim.item():.4f}")

print("\nExercise 4: Normalize matrix rows")
matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
normalized = matrix / matrix.sum(dim=1, keepdim=True)
print(f"Original matrix:\n{matrix}")
print(f"Row-normalized matrix:\n{normalized}")
print(f"Row sums (should all be 1): {normalized.sum(dim=1)}")

print("\nExercise 5: Euclidean distance function")
def euclidean_distance(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2) ** 2))

point_a = torch.tensor([1.0, 2.0, 3.0])
point_b = torch.tensor([4.0, 6.0, 3.0])
dist = euclidean_distance(point_a, point_b)
print(f"Point A: {point_a}")
print(f"Point B: {point_b}")
print(f"Euclidean distance: {dist.item():.4f}")

print("\nPART 5: VISUALIZATION")
print("-------------------")

visual_tensor = torch.linspace(0, 10, 100).reshape(10, 10)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(visual_tensor, cmap='viridis')
plt.colorbar()
plt.title('Original Tensor')

kernel = torch.ones(3, 3) / 9.0
kernel = kernel.view(1, 1, 3, 3)
visual_tensor_expanded = visual_tensor.view(1, 1, 10, 10)

padded = torch.nn.functional.pad(visual_tensor_expanded, (1, 1, 1, 1), 'reflect')
blurred = torch.nn.functional.conv2d(padded, kernel)[0, 0]

plt.subplot(1, 2, 2)
plt.imshow(blurred.detach(), cmap='viridis')
plt.colorbar()
plt.title('Blurred Tensor (Convolution)')

plt.tight_layout()
plt.savefig('tensor_visualization.png')
plt.show()

print("\nCONCEPTUAL QUESTIONS")
print("====================")

print("""
Q1. How do PyTorch tensors differ from NumPy arrays? (at least 3 differences)

  1. GPU acceleration: PyTorch tensors can be moved to a CUDA device (tensor.to('cuda'))
     and operations are executed on the GPU. NumPy arrays are CPU-only.

  2. Automatic differentiation (autograd): Setting requires_grad=True on a tensor
     allows PyTorch to track every operation on it and compute gradients automatically
     via tensor.backward(). NumPy has no such mechanism.

  3. Deep learning integration: PyTorch tensors are the native data type for all
     torch.nn layers, optimisers, and loss functions. NumPy arrays must be converted
     before being used inside a neural network graph.

  4. In-place operations: PyTorch marks in-place operations (e.g. tensor.add_()) as
     potentially unsafe for autograd and may raise RuntimeError. NumPy allows in-place
     operations freely.
""")

print("""
Q2. Broadcasting — own explanation and a new example

  Broadcasting lets PyTorch perform element-wise operations between tensors of
  different (but compatible) shapes without copying data. PyTorch aligns shapes
  from the right: dimensions that are 1 are 'stretched' to match the other tensor.

  Example (not in the lab):
    Subtracting a column vector from every column of a matrix:
      matrix = torch.rand(4, 3)   # shape (4, 3)
      col    = torch.rand(4, 1)   # shape (4, 1)
      result = matrix - col       # col is broadcast across all 3 columns -> (4, 3)
""")

print("""
Q3. Relationship between a tensor's shape and its dimensionality

  A tensor's *dimensionality* (also called rank or number of axes) is simply the
  number of elements in its shape tuple. For example:
    - shape ()     -> 0-D (scalar)
    - shape (5,)   -> 1-D (vector with 5 elements)
    - shape (3, 4) -> 2-D matrix with 3 rows and 4 columns
  The shape fully describes how elements are organised along each axis, while the
  dimensionality tells you how many axes exist.
""")

print("""
Q4. tensor.view() vs tensor.reshape()

  view() requires the tensor to be contiguous in memory; it returns a new tensor that
  shares the exact same data storage — no copy is made, so it is slightly faster and
  uses zero extra memory. If the tensor is not contiguous, view() raises a RuntimeError.

  reshape() works on any tensor: if the data is already contiguous it behaves like
  view() (no copy); otherwise it returns a new contiguous tensor (copy).

  Prefer view() when you know the tensor is contiguous and want to guarantee no copy
  is made (useful inside performance-critical training loops).
""")

print("""
Q5. PyTorch automatic differentiation (autograd)

  When a tensor is created with requires_grad=True, PyTorch records every operation
  applied to it in a dynamic computational graph (DAG). Each node stores a reference
  to the function that produced it (grad_fn). Calling .backward() on a scalar output
  traverses the graph in reverse (chain rule) and accumulates the partial derivatives
  into the .grad attribute of every leaf tensor that has requires_grad=True.

  Example:
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 3          # y = x^3, grad_fn=PowBackward
    y.backward()        # dy/dx = 3x^2 = 12
    print(x.grad)       # tensor([12.])
""")

print("\nEXTENSION ACTIVITIES")
print("====================")

print("\nExtension 1: Linear Regression with gradient computation")

torch.manual_seed(42)
X_train = torch.linspace(0, 1, 20).unsqueeze(1)
y_train = 3.0 * X_train + 2.0 + torch.randn(20, 1) * 0.1

weight = torch.randn(1, requires_grad=True)
bias   = torch.randn(1, requires_grad=True)

lr = 0.5
for epoch in range(200):
    y_pred = X_train * weight + bias
    loss   = ((y_pred - y_train) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        weight -= lr * weight.grad
        bias   -= lr * bias.grad

    weight.grad.zero_()
    bias.grad.zero_()

print(f"  Learned weight: {weight.item():.4f}  (expected ~3.0)")
print(f"  Learned bias  : {bias.item():.4f}  (expected ~2.0)")
print(f"  Final MSE loss: {loss.item():.6f}")

print("\nExtension 2: SVD and Eigenvalue computation")

A = torch.tensor([[4.0, 2.0], [1.0, 3.0]])
U, S, Vh = torch.linalg.svd(A)
print(f"  Matrix A:\n{A}")
print(f"  Singular values: {S}")
print(f"  Reconstruction error: {torch.dist(U @ torch.diag(S) @ Vh, A).item():.6f}")

eigenvalues, eigenvectors = torch.linalg.eig(A)
print(f"  Eigenvalues: {eigenvalues.real}")
print(f"  Eigenvectors:\n{eigenvectors.real}")

print("\nExtension 3: Custom visualisation — sine wave surface")

N = 50
x_vals = torch.linspace(-torch.pi, torch.pi, N)
y_vals = torch.linspace(-torch.pi, torch.pi, N)
grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing='ij')
Z = torch.sin(grid_x) * torch.cos(grid_y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(Z.numpy(), cmap='RdBu', origin='lower',
                     extent=[-3.14, 3.14, -3.14, 3.14])
axes[0].set_title('sin(x) * cos(y)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(Z.abs().numpy(), cmap='plasma', origin='lower',
                     extent=[-3.14, 3.14, -3.14, 3.14])
axes[1].set_title('|sin(x) * cos(y)|')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

plt.suptitle('Extension 3: Sine-Cosine Surface', fontsize=13)
plt.tight_layout()
plt.savefig('extension_visualization.png')
plt.show()
