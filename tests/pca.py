import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def library_pca(X, n_components=None):
    """Perform PCA using scikit-learn"""
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use scikit-learn's PCA
    n_components = min(n_components if n_components is not None else X.shape[1], min(X.shape))
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    # Get components and explained variance
    components = pca.components_.T  # Transpose to match manual_pca output
    explained_variance = pca.explained_variance_
    
    return X_transformed, components, explained_variance

def compare_pca(X, n_components=None, name="Test"):
    """Compare manual and library PCA implementations"""
    print(f"\n=== {name} ===")
    
    # Run manual PCA
    manual_transformed, manual_components, manual_eigvals = manual_pca(X, n_components)
    
    # Run library PCA
    library_transformed, library_components, library_eigvals = library_pca(X, n_components)
    
    # Print manual PCA results
    print("Manual PCA Transformed Data:")
    print("let expected = array![")
    for row in manual_transformed:
        print(f"    [{', '.join([f'{x}' for x in row])}],")
    print("];")
    print(f"\nManual PCA Eigenvalues: {manual_eigvals}")
    
    # Print library PCA results
    print("\nLibrary PCA Transformed Data:")
    print("let expected = array![")
    for row in library_transformed:
        print(f"    [{', '.join([f'{x}' for x in row])}],")
    print("];")
    print(f"\nLibrary PCA Eigenvalues: {library_eigvals}")
    
    # Compare results
    print("\nComparison:")
    # Check if transformed data is similar (allowing for sign flips)
    transformed_similar = np.allclose(np.abs(manual_transformed), np.abs(library_transformed), rtol=1e-5, atol=1e-5)
    print(f"Transformed data similar (accounting for sign flips): {transformed_similar}")
    
    # Compare eigenvalues
    eigvals_similar = np.allclose(manual_eigvals, library_eigvals, rtol=1e-5, atol=1e-5)
    print(f"Eigenvalues similar: {eigvals_similar}")
    
    return transformed_similar and eigvals_similar

# Set print options
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

# Run tests
compare_pca(test1, name="2x2 PCA Comparison")
compare_pca(test2, name="5x7 PCA Comparison")
compare_pca(test2, n_components=4, name="5x7 PCA with k=4 Comparison")

# Random test case with fixed seed
np.random.seed(1926)
rng = np.random.RandomState(1926)
random_2x2 = rng.randn(2, 2)
compare_pca(random_2x2, name="Random 2x2 PCA")

# Additional visualization for understanding differences
print("\n=== Detailed Comparison for 5x7 matrix ===")
manual_tr, manual_comp, manual_eig = manual_pca(test2)
library_tr, library_comp, library_eig = library_pca(test2)

print("\nManual vs Library Eigenvalue Differences:")
for i, (m_eig, l_eig) in enumerate(zip(manual_eig, library_eig)):
    diff = abs(m_eig - l_eig)
    percent = diff / m_eig * 100 if m_eig != 0 else float('inf')
    print(f"Component {i+1}: Manual={m_eig:.8f}, Library={l_eig:.8f}, Diff={diff:.8f}, Percent={percent:.4f}%")

# Check if eigenvectors are similar (allowing for sign flips)
for i in range(min(manual_comp.shape[1], library_comp.shape[1])):
    m_vec = manual_comp[:, i]
    l_vec = library_comp[:, i]
    
    # Check correlation (accounting for possible sign flip)
    corr = abs(np.dot(m_vec, l_vec) / (np.linalg.norm(m_vec) * np.linalg.norm(l_vec)))
    print(f"\nComponent {i+1} eigenvector correlation: {corr:.8f}")
    
    # Print the vectors for comparison
    print(f"Manual:  {m_vec}")
    print(f"Library: {l_vec}")
