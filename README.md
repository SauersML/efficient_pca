# Principal component analysis (PCA)

Forked from https://github.com/ekg/pca. Modified from Erik Garrison's original implementation.

Consider using this library if you have more features than samples.

A Rust library providing **Principal Component Analysis (PCA)** functionality using either:
1. A **covariance-based** eigen-decomposition (classical PCA). (Faster, less memory-efficient.)
2. A **randomized SVD** approach (for large-scale or high-dimensional data). (Slower, more memory-efficient.)

Instead of building a large covariance matrix based on the number of features, which takes a lot of time and memory when features are numerous, it can create a Gram matrix by multiplying the data matrix with its transpose and scaling it. Then, it find eigenvectors of the Gram matrix. Since PCA needs feature-based directions, it transforms these sample-based eigenvectors by multiplying them with the transposed data matrix and dividing by the square root of their eigenvalues, producing the same principal components as the standard method but with less computation because the Gram matrix can be smaller so easier to handle.

This library supports:
- Mean-centering and scaling of input data.
- Automatic selection of PCA components via a user-defined tolerance or a fixed count.
- Both "standard" PCA for moderate dimensions and a "randomized SVD" routine for very large matrices.

[The PCA is obtained via SVD](https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
efficient_pca = "0.1.3"
```

Or just `cargo add efficient_pca` to get the latest version.


## Features

- **Standard PCA** via eigen-decomposition of the covariance matrix:
  - Suitable for data where the number of features is not prohibitively large compared to samples.
  - Automatically handles scaling and centering of your data.
  - Allows a tolerance-based cutoff for the number of principal components.

- **Randomized PCA** (`rfit`) via randomized SVD:
  - Efficient for high-dimensional or very large datasets.
  - Randomized SVD can approximate the principal components much faster if the dimensionality is very large.
  - Allows specifying the number of components and an oversampling parameter to improve accuracy.

- **Flexible Tolerances**:
  - You can specify a fraction of the largest eigenvalue/singular value as a threshold to keep or reject components.

- **Easy Transformation**:
  - Once fitted, the same PCA instance can be used to transform new data into the principal-component space.


## Usage
### Classical PCA (`fit`)

```
use ndarray::array;
use efficient_pca::PCA;

fn main() {
    // Suppose we have some 2D dataset (n_samples x n_features)
    let data = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ];

    // Create a new PCA instance
    let mut pca = PCA::new();

    // Fit the model to the data using the classical covariance-based approach.
    // (Optionally provide a tolerance, e.g., Some(0.01) to drop small components.)
    pca.fit(data.clone(), None).unwrap();

    // Transform the data into the new PCA space
    let transformed = pca.transform(data).unwrap();
    println!("PCA-transformed data:\n{:?}", transformed);
}
```

### Randomized SVD PCA (`rfit`)

```
use ndarray::array;
use efficient_pca::PCA;

fn main() {
    // Larger data matrix example (here just 2x2 for illustration)
    let data = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];

    // Set up PCA
    let mut pca = PCA::new();

    // Use 'rfit' to perform randomized SVD
    // - n_components: number of principal components you want
    // - n_oversamples: oversample dimension
    // - seed: optional for reproducibility
    // - tol: optional variance-based cutoff
    pca.rfit(data.clone(), 2, 10, Some(42), None).unwrap();

    // Transform into PCA space
    let transformed = pca.transform(data).unwrap();
    println!("Randomized PCA result:\n{:?}", transformed);
}
```

### Transforming Data

Once the PCA is fitted (by either `fit` or `rfit`), you can transform new incoming data.  
The PCA object internally stores the mean, scaling, and rotation matrix.  

```
use ndarray::array;
use efficient_pca::PCA;

fn main() {
    let data_train = array![
        [1.0, 2.0],
        [3.0, 4.0],
    ];
    let data_test = array![
        [2.0, 3.0],
        [4.0, 5.0],
    ];
    
    let mut pca = PCA::new();
    pca.fit(data_train.clone(), Some(1e-3)).unwrap();
    
    // Transform both training data and new data
    let train_pcs = pca.transform(data_train).unwrap();
    let test_pcs = pca.transform(data_test).unwrap();
    
    println!("Train set in PCA space:\n{:?}", train_pcs);
    println!("Test set in PCA space:\n{:?}", test_pcs);
}
```


## API Overview

### `PCA::new()`
Creates a new, empty `PCA` struct. Before use, you must call either `fit` or `rfit`.  

```
use efficient_pca::PCA;
let pca = PCA::new();
```

### `PCA::fit(...)`
Fits PCA using the **covariance eigen-decomposition** approach.  
- **Parameters**:
  - `data_matrix`: Your data, shape (n_samples, n_features).
  - `tolerance`: If `Some(tol)`, discard all components whose eigenvalue is below `tol * max_eigenvalue`.
    Otherwise, keep all.  
- **Returns**: `Result<(), Box<dyn Error>>`, but on success, stores internal rotation, mean, and scale.

```
use ndarray::array;
use efficient_pca::PCA;
let data = array![[1.0, 2.0],
                  [3.0, 4.0]];
let mut pca = PCA::new();
pca.fit(data, Some(0.01)).unwrap();
```

### `PCA::rfit(...)`
Fits PCA using **randomized SVD**.  
- **Parameters**:
  - `x`: The input data, shape (n_samples, n_features).
  - `n_components`: Number of components to keep (upper bound).
  - `n_oversamples`: Oversampling dimension for randomized SVD.
  - `seed`: Optional RNG seed for reproducibility.
  - `tol`: Optional fraction of the largest singular value used to drop components.  
- **Returns**: Same as `fit`, but uses a different internal approach optimized for large dimensions.

```
use ndarray::array;
use efficient_pca::PCA;
let data = array![[1.0, 2.0],
                  [3.0, 4.0]];
let mut pca = PCA::new();
pca.rfit(data, 10, 5, Some(42_u64), Some(0.01)).unwrap();
```

### `PCA::transform(...)`
Transforms data using the previously fitted PCA’s rotation, mean, and scale.  
- **Parameters**:
  - `x`: A data matrix with the same number of features as the training data.
- **Returns**: The matrix of shape (n_samples, n_components) in principal-component space.

```
use ndarray::array;
use efficient_pca::PCA;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = array![[1.0, 2.0],
                      [3.0, 4.0]];
    let mut pca = PCA::new();
    pca.fit(data.clone(), None).unwrap();
    let projected = pca.transform(data)?;
    println!("{:?}", projected);
    Ok(())
}
```

## Performance Considerations

**Standard PCA** (`fit`):
   - Computes the covariance matrix `(p x p)` if `p <= n`, else a Gram matrix `(n x n)`.  
   - Faster for smaller feature counts but can be expensive for extremely large `p`.
   
**Randomized SVD** (`rfit`):
   - Best for very high dimensional datasets.
   - You can tune `n_oversamples` to reduce approximation errors (at the cost of more computation).
   - Internally, it uses a smaller SVD on a projected matrix, offering a big speed-up for large or tall/skinny/wide matrices.

**Memory Usage**:
   - The library copies data in some places to center and scale, and creates temporary matrices for covariance or Gram decompositions.
   - For very large datasets, consider using `rfit`.
   - The randomized approach does not actually do streaming now, though it could.
  
- **Use `rfit`** for high-dimensional, wide datasets where memory efficiency is crucial, even if that sometimes means a trade-off in runtime.

## Authors

- Erik Garrison <erik.garrison@gmail.com>. See original repository: https://github.com/ekg/pca
- SauersML

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
