import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fixed dataset
X_py = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
n_components = 3 # min(X_py.shape)

# 1. Scaler (consistent with Rust implementation's scaling of centered data)
X_mean = X_py.mean(axis=0)
X_centered = X_py - X_mean
# Rust PCA calculates std of the mean-centered data with N in denominator (ddof=0)
# This corresponds to X_centered.std(axis=0, ddof=0)
rust_like_scale_factors = X_centered.std(axis=0, ddof=0)

# For sklearn PCA, it standardizes internally by default (with_mean=True, with_std=True)
# which uses X.mean(axis=0) and X.std(axis=0, ddof=0) on the *original* data.
# So, we must use StandardScaler first for a fair comparison to our Rust PCA's specific scaling.
scaler = StandardScaler(with_mean=True, with_std=True) # default ddof=0
X_scaled_for_sklearn = scaler.fit_transform(X_py)
# The actual mean and scale used by this specific StandardScaler instance:
sklearn_scaler_mean = scaler.mean_
sklearn_scaler_scale = scaler.scale_ # This is X.std(axis=0, ddof=0)

# 2. PCA itself
pca = PCA(n_components=n_components, svd_solver='full') # svd_solver='full' is like our Rust fit
pca.fit(X_scaled_for_sklearn)

# Components (n_components, n_features)
# Transpose to match our Rust convention (n_features, n_components)
# Also, sklearn components can have flipped signs compared to other solvers.
# We need to be mindful of this. Let's get them as is first.
components_transposed = pca.components_.T 
explained_variance = pca.explained_variance_ # These are eigenvalues of cov(X_scaled_for_sklearn)

# For our Rust test, the mean and scale are from our Rust PCA's method:
# Mean is X_py.mean(axis=0)
# Scale is std(X_py - X_py.mean(axis=0), ddof=0)

print(f"Data_Mean: {X_mean.tolist()}")
print(f"Data_Scale_For_Rust_PCA: {rust_like_scale_factors.tolist()}")
print(f"Sklearn_Scaler_Mean: {sklearn_scaler_mean.tolist()}")
print(f"Sklearn_Scaler_Scale: {sklearn_scaler_scale.tolist()}")
print(f"PCA_Components_Transposed (n_features x n_components):\n{components_transposed.tolist()}")
print(f"PCA_Explained_Variance: {explained_variance.tolist()}")

# Sanity check: Covariance matrix of X_scaled_for_sklearn
# cov_matrix = np.cov(X_scaled_for_sklearn, rowvar=False, ddof=1) # ddof=1 for sample covariance
# eigvals, eigvecs = np.linalg.eigh(cov_matrix)
# print(f"Manual_Eigvals_from_Cov_np.cov: {np.sort(eigvals)[::-1].tolist()}") # sklearn sorts this
# print(f"Manual_Eigvecs_from_Cov_np.cov:\n{eigvecs[:, np.argsort(eigvals)[::-1]].tolist()}")
