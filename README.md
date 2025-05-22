# Efficient Principal Component Analysis (PCA) in Rust

This Rust library provides Principal Component Analysis (PCA), both exact and fast approximate methods. It is a modified version of the original work by Erik Garrison. Forked from https://github.com/ekg/pca.

---
## Core Features

* ✨ **Exact PCA (`fit`)**: Computes principal components via eigen-decomposition of the covariance matrix. For datasets where the number of features is greater than the number of samples (`n_features > n_samples`), it uses the Gram matrix method (the "Gram trick"). Allows for component selection based on an eigenvalue tolerance.
* ⚡ **Randomized PCA (`rfit`)**: Employs a memory-efficient randomized SVD algorithm to approximate principal components.
* 🛡️ **Data Handling**:
    * Input data is automatically mean-centered.
    * Feature scaling is applied using standard deviations. Scale factors are sanitized to always be positive.
    * Computed principal component vectors are normalized to unit length.
* 💾 **Model Persistence**: Fitted PCA models (including mean, scale factors, and rotation matrix) can be saved to and loaded from files using `bincode` serialization.
* 🔄 **Data Transformation**: Once a PCA model is fitted or loaded, it can be used to transform new data into the principal component space. This transformation also applies the learned centering and scaling.

---
## Installation

Add `efficient_pca` to your `Cargo.toml` dependencies.

```
cargo add efficient_pca
```

---
## API Overview

### `PCA::new()`
Creates a new, empty `PCA` struct. The model is not fitted and needs to be computed using `fit` or `rfit`, or loaded.

### `PCA::with_model(rotation, mean, raw_standard_deviations)`
Creates a `PCA` instance from pre-computed components (rotation matrix, mean vector, and raw standard deviations).
* `raw_standard_deviations`: Input standard deviations for each feature. Values `s` that are non-finite or where `s <= 1e-9` are sanitized to `1.0` before being stored. This makes sure the internal scale factors are always positive and finite. An error is returned if input `raw_standard_deviations` initially contains non-finite values.

### `PCA::fit(&mut self, data_matrix, tolerance)`
Computes the PCA model using an exact method.
* `data_matrix`: The input data (`n_samples` x `n_features`).
* `tolerance`: Optional `f64`. If `Some(tol_val)`, principal components corresponding to eigenvalues less than `tol_val * max_eigenvalue` are discarded. `tol_val` is clamped to `[0.0, 1.0]`. If `None`, all components up to the matrix rank are retained.

### `PCA::rfit(&mut self, x_input_data, n_components_requested, n_oversamples, seed, tol)`
Computes an approximate PCA model using a memory-efficient randomized SVD algorithm and returns the transformed principal component scores for `x_input_data`.
* `x_input_data`: The input data (`n_samples` x `n_features`). This matrix is modified in place for centering and scaling.
* `n_components_requested`: The target number of principal components to compute and keep.
* `n_oversamples`: Number of additional random dimensions (`p`) to sample for the sketch (`l = k + p`).
    * If `0`, an adaptive default for `p` is used (typically 10% of `n_components_requested`, clamped between 5 and 20).
    * If positive, this value is used, but an internal minimum is enforced for robustness. Recommended explicit values: 5-20.
* `seed`: Optional `u64` for the random number generator.
* `tol`: Optional `f64` (typically between 0.0 and 1.0, exclusive). If `Some(t_val)` where `0.0 < t_val < 1.0`, components are further filtered if their corresponding singular value `s_i` from the internal SVD of the projected sketch satisfies `s_i <= t_val * s_max`.

### `PCA::transform(&self, x)`
Applies the learned PCA transformation (centering, scaling, and rotation) to new data `x`.
* `x`: Input data to transform (`m_samples` x `d_features`). This matrix is modified in place during centering and scaling.

### `PCA::rotation(&self)`
Returns an `Option<&Array2<f64>>` to the rotation matrix (principal components), if computed. Shape: (`n_features`, `k_components`).

### `PCA::explained_variance(&self)`
Returns an `Option<&Array1<f64>>` to the explained variance for each principal component, if computed.

### `PCA::save_model(&self, path)`
Saves the current PCA model (rotation, mean, scale, and optionally explained_variance) to the specified file path using `bincode` serialization.

### `PCA::load_model(path)`
Loads a PCA model from a file previously saved with `save_model`. The loaded model is validated for completeness and internal consistency (e.g., matching dimensions, positive scale factors).

---
## Performance Considerations

* **`fit()`**: Provides exact PCA. It's generally suitable for datasets where the smaller dimension (either samples or features) is not excessively large, allowing for direct eigen-decomposition. It automatically uses the Gram matrix optimization if `n_features > n_samples`.
* **`rfit()`**: A significant speed-up and reduced memory footprint for very large or high-dimensional datasets where an approximation of PCA is acceptable. The accuracy is typically good.

---
## Authors and Acknowledgements

* This library is a fork and modification of the original `pca` crate by Erik Garrison (original repository: <https://github.com/ekg/pca>).
* Extended by SauersML.
---
## License

This project is licensed under the MIT License.
