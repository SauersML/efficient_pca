# Principal component analysis (PCA)

Forked from https://github.com/ekg/pca. Modified from Erik Garrison's original implementation.

This Rust library provides Principal Component Analysis (PCA) using two main approaches:
1.  **Exact PCA (`fit`)**: Uses eigen-decomposition of the covariance matrix. If the number of features exceeds the number of samples, it efficiently uses a Gram matrix computation (the "Gram trick").
2.  **Randomized PCA (`rfit`)**: Employs a randomized SVD algorithm, suitable for approximating PCA on very large or high-dimensional datasets.

All computed principal component vectors are normalized to unit length. The library handles mean-centering and ensures data is scaled by positive factors derived from standard deviations.


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
efficient_pca = "0.1.4"
```

Or use `cargo add efficient_pca` to add the latest version.

## Core Features

-   **Exact PCA (`fit`)**: Computes principal components via eigen-decomposition. Handles high-dimensional data (`n_features > n_samples`) efficiently using the Gram matrix method. Allows component selection based on an eigenvalue tolerance.
-   **Randomized PCA (`rfit`)**: Approximates principal components using randomized SVD, designed for very large datasets. Users specify the target number of components and can optionally use a singular value tolerance.
-   **Robust Data Scaling**: Input data is automatically centered. Scaling factors are derived from standard deviations and are always positive (values from original standard deviations `s` where `s.abs() < 1e-9` are sanitized to `1.0`; for `with_model`, inputs `s <= 1e-9` or non-finite also become `1.0`).
-   **Model Persistence**: Fitted PCA models can be saved to and loaded from files. Loaded models are validated for consistency and positive scale factors.
-   **Data Transformation**: Fitted models can transform new data into the principal component space.

(Note: Some tests in the repository require Python and scikit-learn for comparison.)

## Usage

### Basic Workflow (using `fit`)

```rust
use ndarray::array;
use efficient_pca::PCA;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = array![
        [1.0, 2.0, 0.5],
        [3.0, 4.0, 1.5],
        [5.0, 6.0, 2.5],
        [7.0, 8.0, 3.5]
    ];

    let mut pca = PCA::new();

    // Fit the model (e.g., keep all components)
    pca.fit(data.clone(), None)?;

    // Transform data
    let transformed_data = pca.transform(data)?;
    println!("PCA-transformed data:\n{:?}", transformed_data);

    // Example of saving and loading
    let model_path = "pca_model.bin";
    pca.save_model(model_path)?;
    let loaded_pca = PCA::load_model(model_path)?;
    let transformed_by_loaded = loaded_pca.transform(array![[0.0, 1.0, 2.0]])?;
    println!("Transformed by loaded model:\n{:?}", transformed_by_loaded);
    // std::fs::remove_file(model_path)?; // Clean up example file

    Ok(())
}
```

## API Overview

### `PCA::new()`
Creates an empty `PCA` model.

### `PCA::with_model(rotation, mean, raw_standard_deviations)`
Creates a `PCA` model from pre-computed components.
-   `raw_standard_deviations`: Input standard deviations. Values `s` where `s.is_finite() == false` or `s <= 1e-9` are sanitized to `1.0`. Errors if non-finite values are present before this sanitization.

### `PCA::fit(data_matrix, tolerance)`
Fits PCA using an exact method (covariance or Gram matrix).
-   `tolerance`: Optional `f64`. If `Some(tol)`, components with eigenvalue `< tol * max_eigenvalue` are discarded. `tol` is clamped to `[0.0, 1.0]`. If `None`, all components are kept.

### `PCA::rfit(x, n_components, n_oversamples, seed, tol)`
Fits PCA using randomized SVD.
-   `n_components`: Target number of components (upper bound).
-   `n_oversamples`: For randomized algorithm stability.
-   `seed`: Optional `u64` for reproducibility.
-   `tol`: Optional `f64`. If `Some(t_val)` where `0.0 < t_val <= 1.0`, further filters components based on singular values.

### `PCA::transform(x)`
Transforms input data `x` using the fitted model. Data is centered and scaled using stored positive scale factors.

### `PCA::save_model(path)`
Saves the fitted model to `path`.

### `PCA::load_model(path)`
Loads a model from `path`. Validates internal consistency and ensures scale factors are positive.

## Performance Notes

-   **`fit()`**: Generally preferred for exact results when the smaller dimension (`n_samples` or `n_features`) is manageable for eigen-decomposition. Uses Gram matrix for `n_features > n_samples`.
-   **`rfit()`**: More efficient for very large datasets where an approximation is acceptable or exact computation is too costly.

## Authors

-   Erik Garrison <erik.garrison@gmail.com>. Original repository: https://github.com/ekg/pca
-   SauersML

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
