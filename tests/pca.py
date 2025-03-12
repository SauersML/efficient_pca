import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.linalg as la

def manual_pca(X, n_components=None):
    """
    Perform PCA with proper normalization and using the covariance trick
    when n_features > n_samples.
    """
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples, n_features = X_scaled.shape
    
    # Determine number of components if not specified
    if n_components is None:
        n_components = min(n_samples, n_features)
    else:
        n_components = min(n_components, min(n_samples, n_features))
    
    # Apply covariance trick when n_features > n_samples
    if n_features > n_samples:
        # Use the Gram matrix X·X^T (covariance trick)
        gram_matrix = np.dot(X_scaled, X_scaled.T) / (n_samples - 1)
        
        # Eigendecomposition of Gram matrix
        eigvals, eigvecs = la.eigh(gram_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Select the top components
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
        
        # Calculate the actual principal components (eigenvectors of covariance matrix)
        # V = X^T · U / sqrt(λ)
        components = np.zeros((n_features, n_components))
        for i in range(n_components):
            # Handle very small eigenvalues to avoid division by zero
            scale_factor = np.sqrt(eigvals[i]) if eigvals[i] > 1e-12 else 1e-12
            components[:, i] = np.dot(X_scaled.T, eigvecs[:, i]) / (scale_factor * np.sqrt(n_samples - 1))
            # Normalize
            components[:, i] = components[:, i] / np.linalg.norm(components[:, i])
        
        # Transform data
        X_transformed = np.dot(X_scaled, components)
    else:
        # Use standard covariance matrix X^T·X
        cov_matrix = np.dot(X_scaled.T, X_scaled) / (n_samples - 1)
        
        # Eigendecomposition
        eigvals, eigvecs = la.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Limit components
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
        
        # Transform data
        X_transformed = np.dot(X_scaled, eigvecs)
        
        # The eigenvectors are already the components
        components = eigvecs
    
    return X_transformed, components, eigvals

def library_pca(X, n_components=None):
    """Perform PCA using scikit-learn"""
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine max components (can't exceed min(n_samples, n_features))
    max_components = min(X.shape)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)
    
    # Use scikit-learn's PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    # Get components and explained variance
    components = pca.components_.T  # Transpose to match manual_pca output
    explained_variance = pca.explained_variance_
    
    return X_transformed, components, explained_variance

def print_pca_results(X, n_components=None, name="Test"):
    """Print PCA results for a given input and number of components"""
    print(f"\n=== {name} ===")
    
    # Run manual PCA
    manual_transformed, manual_components, manual_eigvals = manual_pca(X, n_components)
    
    # Print manual PCA results
    print("Manual PCA Transformed Data:")
    print("let expected = array![")
    for row in manual_transformed:
        print(f"    [{', '.join([f'{x}' for x in row])}],")
    print("];")
    print(f"\nManual PCA Eigenvalues: {manual_eigvals}")
    
    # Return the result for potential further use
    return manual_transformed, manual_components, manual_eigvals

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
    # We need to check column by column since the sign might be flipped
    transformed_similar = True
    for i in range(manual_transformed.shape[1]):
        manual_col = manual_transformed[:, i]
        library_col = library_transformed[:, i]
        
        # Try both signs
        sim_positive = np.allclose(manual_col, library_col, rtol=1e-5, atol=1e-5)
        sim_negative = np.allclose(manual_col, -library_col, rtol=1e-5, atol=1e-5)
        
        if not (sim_positive or sim_negative):
            transformed_similar = False
            break
    
    print(f"Transformed data similar (accounting for sign flips): {transformed_similar}")
    
    # Compare eigenvalues
    eigvals_similar = np.allclose(manual_eigvals, library_eigvals, rtol=1e-5, atol=1e-5)
    print(f"Eigenvalues similar: {eigvals_similar}")
    
    return transformed_similar and eigvals_similar

# Set print options
np.set_printoptions(precision=15)

# Test case for test_pca_2x2
test_2x2 = np.array([
    [0.5855288, -0.1093033],
    [0.7094660, -0.4534972]
])

# Test case for test_pca_3x5
test_3x5 = np.array([
    [0.5855288, -0.4534972, 0.6300986, -0.9193220, 0.3706279],
    [0.7094660, 0.6058875, -0.2761841, -0.1162478, 0.5202165],
    [-0.1093033, -1.8179560, -0.2841597, 1.8173120, -0.7505320]
])

# Test case for test_pca_5x5
test_5x5 = np.array([
    [0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219],
    [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851],
    [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284],
    [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374],
    [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095]
])

# Test case for test_pca_5x7 and test_rpca_5x7_k3
test_5x7 = np.array([
    [0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
    [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
    [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
    [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
    [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]
])

# Print results for test_pca_2x2
print_pca_results(test_2x2, name="test_pca_2x2 Ground Truth")

# Print results for test_pca_3x5
print_pca_results(test_3x5, name="test_pca_3x5 Ground Truth")

# Print results for test_pca_5x5
print_pca_results(test_5x5, name="test_pca_5x5 Ground Truth")

# Print results for test_pca_5x7
print_pca_results(test_5x7, name="test_pca_5x7 Ground Truth")

# Print results for test_rpca_5x7_k3 (with n_components=3)
print_pca_results(test_5x7, n_components=3, name="test_rpca_5x7_k3 Ground Truth")

# Compare with scikit-learn PCA for validation
compare_pca(test_2x2, name="2x2 PCA Comparison")
compare_pca(test_3x5, name="3x5 PCA Comparison")
compare_pca(test_5x5, name="5x5 PCA Comparison")
compare_pca(test_5x7, name="5x7 PCA Comparison")
compare_pca(test_5x7, n_components=3, name="5x7 PCA with k=3 Comparison")

# Additional visualization for understanding differences
print("\n=== Detailed Comparison for 5x7 matrix ===")
manual_tr, manual_comp, manual_eig = manual_pca(test_5x7)
library_tr, library_comp, library_eig = library_pca(test_5x7)

print("\nManual vs Library Eigenvalue Differences:")
for i, (m_eig, l_eig) in enumerate(zip(manual_eig, library_eig)):
    diff = abs(m_eig - l_eig)
    percent = diff / m_eig * 100 if m_eig != 0 else float('inf')
    print(f"Component {i+1}: Manual={m_eig:.8f}, Library={l_eig:.8f}, Diff={diff:.8f}, Percent={percent:.4f}%")

# Check if eigenvectors are similar (allowing for sign flips)
for i in range(min(manual_comp.shape[1], library_comp.shape[1])):
    m_vec = manual_comp[:, i]
    l_vec = library_comp[:, i]
    
    # Calculate correlation coefficient (accounting for possible sign flip)
    corr = abs(np.dot(m_vec, l_vec) / (np.linalg.norm(m_vec) * np.linalg.norm(l_vec)))
    print(f"\nComponent {i+1} eigenvector correlation: {corr:.8f}")
    
    # Print the vectors for comparison
    print(f"Manual:  {m_vec}")
    print(f"Library: {l_vec}")
