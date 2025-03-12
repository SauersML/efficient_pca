import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.linalg as la

def manual_pca(X, n_components=None):
    """Perform PCA with proper normalization"""
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute covariance matrix
    n_samples = X_scaled.shape[0]
    cov_matrix = np.dot(X_scaled.T, X_scaled) / (n_samples - 1)
    
    # Eigendecomposition
    eigvals, eigvecs = la.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Limit components if specified
    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
    
    # Correctly normalize by sqrt(eigenvalues)
    for i in range(len(eigvals)):
        scale_factor = np.sqrt(eigvals[i]) if eigvals[i] > 1e-12 else 1e-12
        eigvecs[:, i] /= scale_factor
    
    # Transform data
    X_transformed = np.dot(X_scaled, eigvecs)
    
    return X_transformed, eigvecs, eigvals

np.set_printoptions(precision=15)

# Test case 1: 2x2 matrix
test1 = np.array([
    [0.5855288, -0.1093033],
    [0.7094660, -0.4534972]
])

# Test case 2: 5x7 matrix 
test2 = np.array([
    [0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
    [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
    [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
    [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
    [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]
])

# Run tests and print Rust-formatted results
print("=== Correct values for 2x2 PCA ===")
tr1, components1, eigvals1 = correct_pca(test1)
print("let expected = array![")
for row in tr1:
    print(f"    [{', '.join([f'{x}' for x in row])}],")
print("];")
print(f"\nEigenvalues: {eigvals1}")

print("\n=== Correct values for 5x7 PCA ===")
tr2, components2, eigvals2 = correct_pca(test2)
print("let expected = array![")
for row in tr2:
    print(f"    [{', '.join([f'{x}' for x in row])}],")
print("];")
print(f"\nEigenvalues: {eigvals2}")

# RPCA with 4 components
print("\n=== Correct values for RPCA 5x7 k=4 ===")
tr3, components3, eigvals3 = correct_pca(test2, n_components=4)
print("let expected = array![")
for row in tr3:
    print(f"    [{', '.join([f'{x}' for x in row])}],")
print("];")
print(f"\nEigenvalues: {eigvals3}")

np.random.seed(1926)
rng = np.random.RandomState(1926)
random_2x2 = rng.randn(2, 2)
print("\n=== Correct values for RPCA 2x2 with seed 1926 ===")
tr4, components4, eigvals4 = correct_pca(random_2x2)
print("let expected = array![")
for row in tr4:
    print(f"    [{', '.join([f'{x}' for x in row])}],")
print("];")
print(f"\nEigenvalues: {eigvals4}")
