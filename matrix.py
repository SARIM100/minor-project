import numpy as np

def print_matrix(matrix, name):
    print(f"{name}:")
    print(matrix)
    print()

# Create two matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print_matrix(A, "Matrix A")
print_matrix(B, "Matrix B")

# Addition
C = A + B
print_matrix(C, "Matrix Addition (A + B)")

# Subtraction
D = A - B
print_matrix(D, "Matrix Subtraction (A - B)")

# Multiplication
E = A @ B  # or np.matmul(A, B)
print_matrix(E, "Matrix Multiplication (A * B)")

# Transpose
F = A.T
print_matrix(F, "Transpose of A")

# Determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}\n")

# Inverse
try:
    G = np.linalg.inv(A)
    print_matrix(G, "Inverse of A")
except np.linalg.LinAlgError:
    print("Matrix A is not invertible")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A:")
print(eigenvalues)
print("\nEigenvectors of A:")
print(eigenvectors)