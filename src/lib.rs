// Principal component analysis (PCA)

#![doc = include_str!("../README.md")]

use ndarray::{s, Array1, Array2, Axis, ArrayView1};
use ndarray_linalg::{Eigh, QR, SVDInto, UPLO}; // QR is the trait for .qr()
use rand::Rng;
use rand::SeedableRng; // For ChaCha8Rng::seed_from_u64
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal; // Distribution trait is implicitly used by Normal + rng.sample()

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Principal component analysis (PCA) structure.
///
/// This struct holds the results of a PCA (mean, scale, and rotation matrix)
/// and can be used to transform data into the principal component space.
/// It supports both exact PCA computation and a faster, approximate randomized PCA.
/// Models can also be loaded from/saved to files.
#[derive(Serialize, Deserialize, Debug)]
pub struct PCA {
    /// The rotation matrix (principal components).
    /// Shape: (n_features, k_components)
    rotation: Option<Array2<f64>>,
    /// Mean vector of the original training data.
    /// Shape: (n_features)
    mean: Option<Array1<f64>>,
    /// Sanitized scale vector, representing standard deviations of the original training data.
    /// This vector is guaranteed to contain only positive values.
    /// When set via `fit` or `rfit`, original standard deviations `s` where `s.abs() < 1e-9` are replaced by `1.0`.
    /// When set via `with_model`, input `raw_standard_deviations` `s` where `!s.is_finite()` or `s <= 1e-9` are replaced by `1.0`.
    /// Loaded models are also validated so scale factors are positive.
    /// Shape: (n_features)
    scale: Option<Array1<f64>>,
    /// Explained variance for each principal component (eigenvalues of the covariance matrix).
    /// Shape: (k_components)
    explained_variance: Option<Array1<f64>>,
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

impl PCA {
    /// Creates a new, empty PCA struct.
    ///
    /// The PCA model is not fitted and needs to be computed using `fit` or `rfit`,
    /// or loaded using `load_model` or `with_model`.
    ///
    /// # Examples
    ///
    /// ```
    /// use efficient_pca::PCA; // Assuming efficient_pca is your crate name
    /// let pca = PCA::new();
    /// ```
    pub fn new() -> Self {
        Self {
            rotation: None,
            mean: None,
            scale: None,
            explained_variance: None,
        }
    }

    /// Creates a new PCA instance from a pre-computed model.
    ///
    /// This is useful for loading a PCA model whose components (rotation matrix,
    /// mean, and original standard deviations) were computed externally or
    /// previously. The library will sanitize the provided standard deviations
    /// for consistent scaling.
    ///
    /// * `rotation` - The rotation matrix (principal components), shape (d_features, k_components).
    /// * `mean` - The mean vector of the original data used to compute the PCA, shape (d_features).
    /// * `raw_standard_deviations` - The raw standard deviation vector of the original data,
    ///                               shape (d_features). Values that are not strictly positive
    ///                               (i.e., `s <= 1e-9`, zero, negative), or are non-finite,
    ///                               will be sanitized to `1.0` before being stored.
    ///                               If the original PCA did not involve scaling (e.g., data was
    ///                               already standardized, or only centering was desired),
    ///                               pass a vector of ones.
    ///
    /// # Errors
    /// Returns an error if feature dimensions are inconsistent or if `raw_standard_deviations`
    /// contains non-finite values (this check is performed before sanitization).
    pub fn with_model(
        rotation: Array2<f64>,
        mean: Array1<f64>,
        raw_standard_deviations: Array1<f64>,
    ) -> Result<Self, Box<dyn Error>> {
        let d_features_rotation = rotation.nrows();
        let k_components = rotation.ncols();
        let d_features_mean = mean.len();
        let d_features_raw_std = raw_standard_deviations.len();

        if !(d_features_rotation == d_features_mean && d_features_mean == d_features_raw_std) {
            if !(d_features_rotation == 0 && k_components == 0 && d_features_mean == 0 && d_features_raw_std == 0) {
                 return Err(format!(
                    "Feature dimensions of rotation ({}), mean ({}), and raw_standard_deviations ({}) must match.",
                    d_features_rotation, d_features_mean, d_features_raw_std
                ).into());
            }
        }
        
        if d_features_rotation == 0 && k_components > 0 {
             return Err("Rotation matrix has 0 features but expects components.".into());
        }

        if raw_standard_deviations.iter().any(|&val| !val.is_finite()) {
            // Explicitly reject non-finite inputs early.
            return Err("raw_standard_deviations contains non-finite (NaN or infinity) values.".into());
        }

        // Sanitize scale factors:
        // All scale factors are positive. Values that are not strictly positive (<= 1e-9),
        // or were non-finite (though checked above), are replaced with 1.0.
        // This aligns `with_model`'s scale handling with `fit`/`rfit` and means `self.scale` is always positive.
        let sanitized_scale_vector = raw_standard_deviations
            .mapv(|val| if val.is_finite() && val > 1e-9 { val } else { 1.0 });

        Ok(Self {
            rotation: Some(rotation),
            mean: Some(mean),
            scale: Some(sanitized_scale_vector),
            explained_variance: None, // Explained variance is not provided by this constructor directly
        })
    }

    /// Returns a reference to the rotation matrix (principal components), if computed.
    ///
    /// The rotation matrix has dimensions (n_features, k_components).
    /// Returns `None` if the PCA model has not been fitted, or if the rotation matrix
    /// is not available (e.g., if fitting resulted in zero components).
    pub fn rotation(&self) -> Option<&Array2<f64>> {
        self.rotation.as_ref()
    }

    /// Returns a reference to the explained variance for each principal component.
    ///
    /// These are the eigenvalues of the covariance matrix of the scaled data,
    /// ordered from largest to smallest.
    /// Returns `None` if the PCA model has not been fitted or if variances are not available.
    pub fn explained_variance(&self) -> Option<&Array1<f64>> {
        self.explained_variance.as_ref()
    }

    /// Fits the PCA model to the data using an exact covariance/Gram matrix approach.
    ///
    /// This method computes the mean, (sanitized) scaling factors, and principal axes (rotation)
    /// via an eigen-decomposition of the covariance matrix (if n_features <= n_samples)
    /// or the Gram matrix (if n_features > n_samples, the "Gram trick").
    /// The resulting principal components (columns of the rotation matrix) are normalized to unit length.
    ///
    /// Note: For very large datasets, `rfit` is generally recommended for better performance.
    ///
    /// * `data_matrix` - Input data as a 2D array, shape (n_samples, n_features).
    /// * `tolerance` - Optional: Tolerance for excluding low-variance components
    ///                 (fraction of the largest eigenvalue). If `None`, all components
    ///                 up to the effective rank of the matrix are kept.
    ///
    /// # Errors
    /// Returns an error if the input matrix has zero dimensions, fewer than 2 samples, or if
    /// matrix operations (like eigen-decomposition) fail.
    pub fn fit(
        &mut self,
        mut data_matrix: Array2<f64>, 
        tolerance: Option<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let n_samples = data_matrix.nrows();
        let n_features = data_matrix.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err("Input data_matrix has zero samples or zero features.".into());
        }
        if n_samples < 2 {
            return Err("Input matrix must have at least 2 samples.".into());
        }

        let mean_vector = data_matrix
            .mean_axis(Axis(0))
            .ok_or("Failed to compute mean of the data.")?;
        self.mean = Some(mean_vector.clone());
        data_matrix -= &mean_vector; 

        let original_std_dev_vector = data_matrix.map_axis(Axis(0), |column| column.std(0.0));
        let sanitized_scale_vector = original_std_dev_vector
            .mapv(|val| if val.abs() < 1e-9 { 1.0 } else { val });
        self.scale = Some(sanitized_scale_vector.clone()); 
        
        let mut scaled_data_matrix = data_matrix; 
        scaled_data_matrix /= &sanitized_scale_vector; 

        if n_features <= n_samples {
            let mut cov_matrix = scaled_data_matrix.t().dot(&scaled_data_matrix);
            cov_matrix /= (n_samples - 1) as f64;

            let (vals, vecs) = cov_matrix
                .eigh(UPLO::Upper)
                .map_err(|e| format!("Eigen decomposition of covariance matrix failed: {}",e))?;

            let mut eig_pairs: Vec<(f64, Array1<f64>)> = vals
                .into_iter()
                .zip(vecs.columns().into_iter().map(|col| col.to_owned()))
                .collect();
            eig_pairs.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let rank_limit = if let Some(tol_val) = tolerance {
                let largest_eigval = eig_pairs.get(0).map_or(0.0, |(v, _)| *v);
                if largest_eigval <= 1e-9 { 
                    0
                } else {
                    let threshold = largest_eigval * tol_val.max(0.0).min(1.0); 
                    eig_pairs.iter().take_while(|(val, _)| *val > threshold).count()
                }
            } else {
                eig_pairs.len() // Keep all components if no tolerance
            };
            let final_rank = std::cmp::min(rank_limit, n_features);

            if final_rank == 0 {
                self.rotation = Some(Array2::zeros((n_features, 0)));
                self.explained_variance = Some(Array1::zeros(0));
            } else {
                let mut top_eigvecs_owned: Vec<Array1<f64>> = Vec::with_capacity(final_rank);
                let mut sorted_eigenvalues: Vec<f64> = Vec::with_capacity(final_rank);
                for i in 0..final_rank {
                    let (eig_val, mut eig_vec) = (eig_pairs[i].0, eig_pairs[i].1.clone());
                    sorted_eigenvalues.push(eig_val.max(0.0)); // non-negative eigenvalues
                    let norm = eig_vec.dot(&eig_vec).sqrt();
                    if norm > 1e-9 {
                        eig_vec.mapv_inplace(|x| x / norm);
                    } else {
                        eig_vec.fill(0.0); 
                    }
                    top_eigvecs_owned.push(eig_vec);
                }
                let views: Vec<ArrayView1<f64>> = top_eigvecs_owned.iter().map(|v| v.view()).collect();
                let rotation_matrix = ndarray::stack(Axis(1), &views)?;
                self.rotation = Some(rotation_matrix);
                self.explained_variance = Some(Array1::from(sorted_eigenvalues));
            }
        } else {
            // Gram trick path
            let mut gram_matrix = scaled_data_matrix.dot(&scaled_data_matrix.t());
            gram_matrix /= (n_samples - 1) as f64;

            let (vals, u_vecs) = gram_matrix
                .eigh(UPLO::Upper)
                .map_err(|e| format!("Eigen decomposition of Gram matrix failed: {}", e))?;

            let mut eig_pairs: Vec<(f64, Array1<f64>)> = vals
                .into_iter()
                .zip(u_vecs.columns().into_iter().map(|col| col.to_owned()))
                .collect();
            eig_pairs.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            
            let rank_limit = if let Some(tol_val) = tolerance {
                 let largest_eigval = eig_pairs.get(0).map_or(0.0, |(v, _)| *v);
                if largest_eigval <= 1e-9 {
                    0
                } else {
                    let threshold = largest_eigval * tol_val.max(0.0).min(1.0);
                    eig_pairs.iter().take_while(|(val, _)| *val > threshold).count()
                }
            } else {
                eig_pairs.len() // Keep all components if no tolerance
            };
            let final_rank = std::cmp::min(rank_limit, n_samples); // Rank is capped by n_samples here

            if final_rank == 0 {
                self.rotation = Some(Array2::zeros((n_features, 0)));
                self.explained_variance = Some(Array1::zeros(0));
            } else {
                let mut rotation_matrix = Array2::<f64>::zeros((n_features, final_rank));
                let mut sorted_eigenvalues: Vec<f64> = Vec::with_capacity(final_rank);
                for i in 0..final_rank {
                    let (eigval, ref u_col) = eig_pairs[i];
                    sorted_eigenvalues.push(eigval.max(0.0)); // Store the eigenvalue (variance)
                    // Eigenvalue from G = X'X'^T / (N-1)
                    // V_k = (X'^T u_k) / sqrt(eigval_k_gram * (N-1)) should yield unit norm V_k.
                    
                    // Clamp eigenvalues to a small positive number for calculations to prevent division by zero or very small numbers.
                    let eigval_clamped_for_calc = eigval.max(1e-12);
                    let lam_sqrt = eigval_clamped_for_calc.sqrt();

                    let mut axis_i = scaled_data_matrix.t().dot(u_col);
                    
                    // Denominator for scaling axis_i.
                    // Since n_samples >= 2 and lam_sqrt >= sqrt(1e-12), denom will be non-zero.
                    let denom = lam_sqrt * ((n_samples - 1) as f64).sqrt();
                    if denom.abs() > 1e-9 { // Avoid division by zero if lam_sqrt is extremely small
                        axis_i.mapv_inplace(|x| x / denom);
                    } else {
                        axis_i.fill(0.0); // If denominator is zero, axis is likely zero or ill-defined
                    }
                    
                    // Explicitly re-normalize the principal axis to unit length.
                    // Standard PCA components are unit vectors.
                    let norm_val = axis_i.dot(&axis_i).sqrt();
                    if norm_val > 1e-9 { // Avoid division by zero/small norm if axis_i is effectively zero
                        axis_i.mapv_inplace(|x| x / norm_val);
                    } else {
                        // If the axis vector is effectively zero, represent it as such.
                        axis_i.fill(0.0);
                    }
                    rotation_matrix.slice_mut(s![.., i]).assign(&axis_i);
                }
                self.rotation = Some(rotation_matrix);
                self.explained_variance = Some(Array1::from(sorted_eigenvalues));
            }
        }
        Ok(())
    }

    /// Fits the PCA model using a memory-efficient randomized SVD approach and returns the
    /// transformed principal component scores.
    ///
    /// This method computes the mean of the input data, (sanitized) feature-wise scaling factors
    /// (standard deviations), and an approximate rotation matrix (principal components).
    /// It is specifically designed for computational efficiency and reduced memory footprint
    /// when working with large datasets, particularly those with a very large number of features,
    /// as it avoids forming the full covariance matrix.
    ///
    /// The core of this method is a randomized SVD algorithm that constructs a low-rank
    /// approximation of the input data. It adaptively chooses its sketching strategy based
    /// on the dimensions of the input data matrix `A` (n_samples × n_features, after
    /// centering and scaling):
    ///
    /// - If `n_features <= n_samples` (data matrix is tall or square):
    ///   The algorithm directly sketches the input matrix `A` by forming `Y = AΩ'`, where `Ω'`
    ///   is a random Gaussian matrix of shape (n_features × l_sketch_components).
    ///   An orthonormal basis `Q'` for the range of `Y` is found (N × l_sketch_components).
    ///   The data is then projected onto this basis: `B' = (Q')^T A` (l_sketch_components × n_features).
    ///
    /// - If `n_features > n_samples` (data matrix is wide, D > N):
    ///   To handle a large number of features efficiently, the algorithm sketches the
    ///   transpose `A^T`. It computes `Y = A^T Ω` (n_features × l_sketch_components),
    ///   where `Ω` is a random Gaussian matrix (n_samples × l_sketch_components).
    ///   An orthonormal basis `Q` for the range of `Y` is found (n_features × l_sketch_components).
    ///   The data is then projected: `B = Q^T A^T = (AQ)^T` (l_sketch_components × n_samples).
    ///   *Note: The formation of `Y` (and subsequent refinement of `Q`) is performed
    ///   memory-efficiently by leveraging BLAS capabilities for transposed operations
    ///   on `A` without explicit large-scale copying of `A^T`.*
    ///
    /// In both cases, power iterations are used to refine the orthonormal basis (`Q'` or `Q`)
    /// for improved accuracy.
    ///
    /// An SVD is then performed on the smaller projected matrix (`B'` or `B`), which has dimensions
    /// (l_sketch_components × n_features) or (l_sketch_components × n_samples) respectively.
    /// This step is analogous to the "Gram trick" in exact PCA, significantly reducing
    /// computational cost.
    ///
    /// The principal components (columns of the rotation matrix, stored in `self.rotation`)
    /// are derived from this SVD and are normalized to unit length.
    /// The number of components kept can be influenced by the `tol` (tolerance) parameter.
    ///
    /// The method stores the computed `mean`, `scale` (sanitized standard deviations),
    /// `rotation` matrix, and `explained_variance` within the `PCA` struct instance.
    ///
    /// * `x_input_data` - Input data as a 2D array with shape (n_samples, n_features).
    ///   This matrix will be consumed and its data modified in place for mean centering
    ///   and scaling.
    /// * `n_components_requested` - The target number of principal components to compute and keep.
    ///   The actual number of components kept may be less if the data's effective rank is lower
    ///   or if `tol` filters out components.
    /// * `n_oversamples` - Number of additional random dimensions to sample during the sketching
    ///   phase. This helps improve the accuracy of the randomized SVD. A typical value is 10-20.
    /// * `seed` - Optional `u64` seed for the random number generator used in sketching, allowing
    ///   for reproducible results. If `None`, a random seed is used.
    /// * `tol` - Optional tolerance (a float between 0.0 and 1.0, exclusive of 0.0 if used for filtering).
    ///   If `Some(t_val)`, components are kept if their corresponding singular value `s_i` from the
    ///   internal SVD satisfies `s_i > t_val * s_max`, where `s_max` is the largest singular value.
    ///   The number of components kept will be at most `n_components_requested`.
    ///   If `None`, tolerance-based filtering based on singular value ratios is skipped, and up to
    ///   `n_components_requested` components (or the effective rank) are kept.
    ///
    /// # Returns
    /// A `Result` containing:
    /// - `Ok(Array2<f64>)`: The transformed data (principal component scores) of shape
    ///   (n_samples, k_components_kept), where `k_components_kept` is the
    ///   actual number of principal components retained after all filtering and rank considerations.
    /// - `Err(Box<dyn Error>)`: If an error occurs during the process.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The input matrix `x_input_data` has zero samples or zero features.
    /// - The number of samples `n_samples` is less than 2.
    /// - `n_components_requested` is 0.
    /// - Internal matrix operations (like QR decomposition or SVD) fail.
    /// - Random number generation fails.
    pub fn rfit(
        &mut self,
        mut x_input_data: Array2<f64>, // N x D (n_samples x n_features)
        n_components_requested: usize,
        n_oversamples: usize,
        seed: Option<u64>,
        tol: Option<f64>,
    ) -> Result<Array2<f64>, Box<dyn Error>> {
        let n_samples = x_input_data.nrows();
        let n_features = x_input_data.ncols();

        // --- 1. Input Validations ---
        if n_samples == 0 || n_features == 0 {
            return Err("Input matrix has zero samples or zero features.".into());
        }
        if n_samples < 2 {
            return Err("Input matrix must have at least 2 samples for PCA.".into());
        }
        if n_components_requested == 0 {
            return Err(
                "Number of requested components (n_components_requested) must be greater than 0."
                    .into(),
            );
        }

        // --- 2. Mean Centering and Scaling ---
        let mean_vector = x_input_data
            .mean_axis(Axis(0))
            .ok_or("Failed to compute mean vector.")?;
        self.mean = Some(mean_vector.clone());
        x_input_data -= &mean_vector; // Center data in place.

        let std_dev_vector_original = x_input_data.map_axis(Axis(0), |column| column.std(0.0));
        let sanitized_scale_vector = std_dev_vector_original
            .mapv(|val| if val.abs() < 1e-9 { 1.0 } else { val });
        self.scale = Some(sanitized_scale_vector.clone());
        x_input_data /= &sanitized_scale_vector; // Scale data in place.

        let centered_scaled_data_a = x_input_data; // N x D

        // --- 3. Determine Target Components and Sketch Size ---
        // k_target_components is the n_components_requested, capped by data dimensions later if needed.
        let max_possible_rank = std::cmp::min(n_samples, n_features);

        if max_possible_rank == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
            return Ok(Array2::zeros((n_samples, 0)));
        }

        let l_sketch_components_ideal = n_components_requested + n_oversamples;
        let mut l_sketch_components = std::cmp::min(l_sketch_components_ideal, max_possible_rank);
        l_sketch_components = std::cmp::max(l_sketch_components, 1);
        // sketch size is at least the number of requested components (if possible)
        l_sketch_components = std::cmp::max(
            l_sketch_components,
            std::cmp::min(n_components_requested, max_possible_rank),
        );
        
        // If after all adjustments, l_sketch_components ended up being 0 (e.g. if n_components_requested was 0 and max_possible_rank also 0)
        // though n_components_requested > 0 is checked earlier, and max_possible_rank > 0 is also checked.
        if l_sketch_components == 0 { 
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
            return Ok(Array2::zeros((n_samples, 0)));
        }


        // --- 4. Initialize RNG ---
        const N_POWER_ITERATIONS: usize = 2;
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(rand::thread_rng())
                .map_err(|e| format!("Failed to initialize RNG: {}", e))?,
        };

        // --- 5. Adaptive Randomized SVD Core ---
        let final_rotation_sketch: Array2<f64>; // Will store D x k_eff_l final rotation basis
        let singular_values_from_projected_b: Array1<f64>;

        if n_features <= n_samples {
            // --- Strategy 1: D <= N (Fewer features than samples, or D=N) ---
            let omega_prime_shape = (n_features, l_sketch_components); // D x L
            let omega_prime_random_matrix =
                Array2::from_shape_fn(omega_prime_shape, |_| {
                    rng.sample(Normal::new(0.0, 1.0).unwrap())
                });

            let mut q_prime_basis = centered_scaled_data_a.dot(&omega_prime_random_matrix); // (N x D) . (D x L) -> N x L
            if q_prime_basis.ncols() == 0 {
                return Err("Sketch Q'_basis (from A) has zero columns before QR.".into());
            }
            q_prime_basis = q_prime_basis
                .qr()
                .map_err(|e| format!("QR decomposition of Q' failed: {}", e))?
                .0; // N x L (Q factor)

            for _ in 0..N_POWER_ITERATIONS {
                if q_prime_basis.ncols() == 0 { break; }
                let w_prime_intermediate = centered_scaled_data_a.t().dot(&q_prime_basis); // (D x N view) . (N x L) -> D x L
                if w_prime_intermediate.ncols() == 0 { break; }
                let w_prime_ortho_basis = w_prime_intermediate
                    .qr()
                    .map_err(|e| format!("QR decomposition of W' failed: {}", e))?
                    .0; // D x L

                if w_prime_ortho_basis.ncols() == 0 { break; }
                let z_prime_intermediate = centered_scaled_data_a.dot(&w_prime_ortho_basis); // (N x D) . (D x L) -> N x L
                if z_prime_intermediate.ncols() == 0 { break; }
                q_prime_basis = z_prime_intermediate
                    .qr()
                    .map_err(|e| format!("QR decomposition of Z' failed: {}", e))?
                    .0; // N x L
            }

            if q_prime_basis.ncols() == 0 {
                return Err(
                    "Refined sketch Q'_basis (from A) has zero columns after power iterations."
                        .into(),
                );
            }

            let b_prime_projected = q_prime_basis.t().dot(&centered_scaled_data_a); // (L x N view) . (N x D) -> L x D
            let svd_result_b_prime = b_prime_projected
                .svd_into(false, true) 
                .map_err(|e| format!("SVD of B' (from A sketch) failed: {}", e))?;

            singular_values_from_projected_b = svd_result_b_prime.1;
            let vt_b_prime_matrix = svd_result_b_prime
                .2
                .ok_or("SVD V_B'^T not computed from B' (A sketch)")?; // k_eff_l x D (or L_actual x D)

            final_rotation_sketch = vt_b_prime_matrix.t().into_owned(); // D x k_eff_l
        } else {
            // --- Strategy 2: D > N (More features than samples) ---
            // centered_scaled_data_a (A_orig) is N x D.
            let omega_shape = (n_samples, l_sketch_components); // N x L
            let omega_random_matrix =
                Array2::from_shape_fn(omega_shape, |_| {
                    rng.sample(Normal::new(0.0, 1.0).unwrap())
                });

            // Q_basis = A_orig^T * Omega (D x L)
            let mut q_basis = centered_scaled_data_a.t().dot(&omega_random_matrix); // (D x N view) . (N x L) -> D x L
            if q_basis.ncols() == 0 {
                return Err(
                    "Initial sketch Q_basis (from A_orig^T) has zero columns before QR.".into(),
                );
            }
            q_basis = q_basis
                .qr()
                .map_err(|e| format!("QR decomposition of Q (A_orig^T sketch) failed: {}", e))?
                .0; // D x L (Q factor)

            for _ in 0..N_POWER_ITERATIONS {
                if q_basis.ncols() == 0 { break; }
                // W_intermediate = A_orig * Q_basis (N x L)
                let w_intermediate = centered_scaled_data_a.dot(&q_basis); // (N x D) . (D x L) -> N x L
                if w_intermediate.ncols() == 0 { break; }
                let w_ortho_basis = w_intermediate
                    .qr()
                    .map_err(|e| format!("QR decomposition of W (A_orig^T sketch) failed: {}", e))?
                    .0; // N x L

                if w_ortho_basis.ncols() == 0 { break; }
                // Z_intermediate = A_orig^T * W_ortho_basis (D x L)
                let z_intermediate = centered_scaled_data_a.t().dot(&w_ortho_basis); // (D x N view) . (N x L) -> D x L
                if z_intermediate.ncols() == 0 { break; }
                q_basis = z_intermediate
                    .qr()
                    .map_err(|e| format!("QR decomposition of Z (A_orig^T sketch) failed: {}", e))?
                    .0; // D x L
            }

            if q_basis.ncols() == 0 {
                return Err(
                    "Refined sketch Q_basis (from A_orig^T) has zero columns after power iterations."
                        .into(),
                );
            }
            
            // B_projected^T = A_orig * Q_basis (N x L)
            let b_projected_transpose = centered_scaled_data_a.dot(&q_basis); // N x L
            let b_projected = b_projected_transpose.t().into_owned(); // L x N (owned for SVD)

            let svd_result_b = b_projected
                .svd_into(true, false) // compute_u=true, compute_v=false
                .map_err(|e| format!("SVD of B_projected (from A_orig^T sketch) failed: {}", e))?;

            singular_values_from_projected_b = svd_result_b.1; // k_eff_l elements
            let u_b_matrix = svd_result_b
                .0
                .ok_or("SVD U_B not computed from B_projected (A_orig^T sketch)")?; // L x k_eff_l
            
            // Rotation = Q_basis * U_B_matrix
            final_rotation_sketch = q_basis.dot(&u_b_matrix); // (D x L) . (L x k_eff_l) -> D x k_eff_l
        }

        // --- 6. Post-Sketching: Normalize, Select Components, Update Model ---
        let mut rotation_matrix_normalized = final_rotation_sketch; // D x k_eff_l

        if rotation_matrix_normalized.ncols() == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
            return Ok(Array2::zeros((n_samples, 0)));
        }

        for mut column_vec in rotation_matrix_normalized.columns_mut() {
            let norm_value = column_vec.dot(&column_vec).sqrt();
            if norm_value > 1e-9 {
                column_vec.mapv_inplace(|val| val / norm_value);
            } else {
                column_vec.fill(0.0);
            }
        }

        let k_eff_from_svd_path = rotation_matrix_normalized.ncols(); 
        let mut n_components_to_keep =
            std::cmp::min(n_components_requested, k_eff_from_svd_path);

        if let Some(tolerance_value) = tol {
            if !singular_values_from_projected_b.is_empty()
                && tolerance_value > 0.0
                && tolerance_value <= 1.0
            {
                let largest_singular_value = singular_values_from_projected_b[0];
                if largest_singular_value > 1e-9 {
                    let singular_value_threshold = tolerance_value * largest_singular_value;
                    let rank_by_tolerance = singular_values_from_projected_b
                        .iter()
                        .take_while(|&&s_val| s_val > singular_value_threshold)
                        .count();
                    n_components_to_keep = std::cmp::min(n_components_to_keep, rank_by_tolerance);
                } else {
                    n_components_to_keep = 0;
                }
            }
        }
        n_components_to_keep = std::cmp::max(n_components_to_keep, 0);


        if n_components_to_keep == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
        } else {
            let final_rotation_matrix = rotation_matrix_normalized
                .slice(s![.., ..n_components_to_keep])
                .to_owned();
            self.rotation = Some(final_rotation_matrix);

            if n_samples > 1 {
                let selected_singular_values =
                    singular_values_from_projected_b.slice(s![..n_components_to_keep]);
                let variances = selected_singular_values
                    .mapv(|s_val| s_val.powi(2) / ((n_samples - 1) as f64));
                self.explained_variance = Some(variances);
            } else {
                self.explained_variance =
                    Some(Array1::from_elem(n_components_to_keep, f64::NAN));
            }
        }

        // --- 7. Calculate and Return Principal Component Scores for the Input Data ---
        let rotation_for_transform = self
            .rotation
            .as_ref()
            .ok_or_else(|| "Rotation matrix not set after fitting, this should not happen.")?;
        
        let transformed_principal_components =
            centered_scaled_data_a.dot(rotation_for_transform);

        Ok(transformed_principal_components)
    }

    /// Applies the PCA transformation to the given data.
    ///
    /// The data is centered and scaled using the mean and scale factors
    /// learned during fitting (or loaded into the model), and then projected
    /// onto the principal components.
    ///
    /// * `x` - Input data to transform, shape (m_samples, d_features).
    ///         Can be a single sample (1 row).
    ///
    /// # Errors
    /// Returns an error if the PCA model is not fitted/loaded, or if the input
    /// data's feature dimension does not match the model's.
    pub fn transform(&self, mut x: Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let rotation_matrix = self.rotation.as_ref()
            .ok_or_else(|| "PCA model: Rotation matrix not set. Fit or load a model first.")?;
        let mean_vector = self.mean.as_ref()
            .ok_or_else(|| "PCA model: Mean vector not set. Fit or load a model first.")?;
        let scale_vector = self.scale.as_ref()
            .ok_or_else(|| "PCA model: Scale vector not set. Fit or load a model first.")?;

        let n_input_samples = x.nrows();
        let n_input_features = x.ncols();
        let n_model_features = mean_vector.len();

        if n_input_features != n_model_features {
            return Err(format!(
                "Input data feature dimension ({}) does not match model's feature dimension ({}).",
                n_input_features, n_model_features
            ).into());
        }
        if rotation_matrix.nrows() != n_model_features {
            return Err(format!(
                "Model inconsistency: Rotation matrix feature dimension ({}) does not match model's feature dimension ({}).",
                rotation_matrix.nrows(), n_model_features
            ).into());
        }
        if scale_vector.len() != n_model_features {
             return Err(format!(
                "Model inconsistency: Scale vector dimension ({}) does not match model's feature dimension ({}).",
                scale_vector.len(), n_model_features
            ).into());
        }
        
        if n_input_samples == 0 { 
            let k_components = rotation_matrix.ncols();
            return Ok(Array2::zeros((0, k_components)));
        }

        x -= mean_vector;
        x /= scale_vector;
        Ok(x.dot(rotation_matrix))
    }

    /// Saves the current PCA model to a file using bincode.
    ///
    /// The model must contain rotation, mean, and scale components for saving.
    /// The `explained_variance` field can be `None` (e.g., if the model was created
    /// via `with_model` and eigenvalues were not supplied).
    ///
    /// * `path` - The file path to save the model to.
    ///
    /// # Errors
    /// Returns an error if essential model components (rotation, mean, scale) are missing,
    /// or if file I/O or serialization fails.
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        // Rotation, mean, and scale are essential for a model to be usable for transformation.
        if self.rotation.is_none() || self.mean.is_none() || self.scale.is_none() {
            return Err("Cannot save a PCA model that is missing essential components (rotation, mean, or scale).".into());
        }
        // explained_variance being None is acceptable, for example, if the model was created
        // using `with_model` and eigenvalues were not provided or computed.
        // `load_model` contains further validation for consistency if explained_variance is Some.
        let file = File::create(path.as_ref())
            .map_err(|e| format!("Failed to create file at {:?}: {}", path.as_ref(), e))?;
        let mut writer = BufWriter::new(file); 
        
        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| format!("Failed to serialize PCA model: {}", e))?;
        Ok(())
    }

    /// Loads a PCA model from a file previously saved with `save_model`.
    ///
    /// * `path` - The file path to load the model from.
    ///
    /// # Errors
    /// Returns an error if file I/O or deserialization fails, or if the
    /// loaded model is found to be incomplete, internally inconsistent (e.g., mismatched dimensions),
    /// or contains non-positive scale factors.
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open file at {:?}: {}", path.as_ref(), e))?;
        let mut reader = BufReader::new(file); 
        
        let pca_model: PCA = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| format!("Failed to deserialize PCA model: {}", e))?;

        let rotation = pca_model.rotation.as_ref().ok_or("Loaded PCA model is missing rotation matrix.")?;
        let mean = pca_model.mean.as_ref().ok_or("Loaded PCA model is missing mean vector.")?;
        let scale = pca_model.scale.as_ref().ok_or("Loaded PCA model is missing scale vector.")?;
        
        let d_rot_features = rotation.nrows();
        let d_mean_features = mean.len();
        let d_scale_features = scale.len();

        if !(d_rot_features == d_mean_features && d_mean_features == d_scale_features) {
            if !(d_rot_features == 0 && rotation.ncols() == 0 && d_mean_features == 0 && d_scale_features == 0) {
                return Err(format!(
                    "Loaded PCA model has inconsistent feature dimensions: rotation_features={}, mean_features={}, scale_features={}",
                    d_rot_features, d_mean_features, d_scale_features
                ).into());
            }
        }
        // Validate that loaded scale factors are positive, aligning with the contract for self.scale.
        // self.scale is expected to store sanitized, positive values (1.0 for original std devs <= 1e-9, else the std dev itself).
        if scale.iter().any(|&val| !val.is_finite() || val <= 0.0) {
            return Err("Loaded PCA model's scale vector contains invalid (non-finite, zero, or negative) values. Scale values must be positive.".into());
        }

        // Validate explained_variance if present
        if let Some(ev) = pca_model.explained_variance.as_ref() {
            if let Some(rot) = pca_model.rotation.as_ref() {
                if ev.len() != rot.ncols() {
                    return Err(format!(
                        "Loaded PCA model has inconsistent dimensions: explained_variance length ({}) does not match rotation matrix number of components ({}).",
                        ev.len(), rot.ncols()
                    ).into());
                }
            } else { // Should not happen if rotation is required for a valid model
                 return Err("Loaded PCA model has explained_variance but no rotation matrix.".into());
            }
            if ev.iter().any(|&val| !val.is_finite() || val < 0.0) { // Variances cannot be negative
                return Err("Loaded PCA model's explained_variance vector contains invalid (non-finite or negative) values.".into());
            }
        }
        // If rotation is Some and has components, but explained_variance is None (e.g. model from `with_model`),
        // this is an acceptable state. The `explained_variance()` accessor will simply return None.
        // If rotation itself is None or has no components (ncols == 0), then explained_variance being None is also consistent.

        Ok(pca_model)
    }
}



// ====================================
// ====================================
// ====================================



#[cfg(test)]
mod genome_tests {
    use super::*;
    use ndarray::{s, Array2, ArrayView1};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::io::Write;
    use std::process::Command;
    use std::time::Instant;
    use sysinfo::System;
    use tempfile::NamedTempFile;

    #[derive(Clone, Debug)]
    struct Variant {
        position: i64,
        genotypes: Vec<Option<Vec<u8>>>,
    }
    
    #[test]
    fn test_genomic_pca_with_haplotypes() {
        println!("Testing PCA with haplotype data similar to genomic application");
        
        // Set up parameters to match real usage
        let n_samples = 44; // 44 samples = 88 haplotypes
        let n_variants = 10000;
        let n_components = 10;
        
        // Generate synthetic variants
        let mut variants = Vec::with_capacity(n_variants);
        
        // Create variants with different allele frequencies
        // Some variants will be common, some rare
        for i in 0..n_variants {
            let position = 1000 + i as i64 * 100;
            let mut genotypes = Vec::with_capacity(n_samples);
            
            // Create some extremely rare variants
            let is_extreme_rare = i < 100; // First 100 variants are extremely rare
            
            for sample_idx in 0..n_samples {
                let pop_group = sample_idx % 4;
                
                let base_prob = if is_extreme_rare {
                    // For extremely rare variants, only a single sample in a single population has it
                    if sample_idx == (i % n_samples) && pop_group == 0 { 0.5 } else { 0.0 }
                } else {
                    // Structure for other variants
                    match i % 10 {
                        0..=1 => if pop_group == 0 { 0.4 } else { 0.05 },
                        2..=3 => if pop_group == 1 { 0.3 } else { 0.02 },
                        4..=6 => if pop_group == 2 { 0.5 } else { 0.08 },
                        _ => if pop_group == 3 { 0.3 } else { 0.02 },
                    }
                };
                
                // Create haplotypes
                let left_allele = if rand::random::<f64>() < base_prob { 1u8 } else { 0u8 };
                let right_allele = if rand::random::<f64>() < base_prob { 1u8 } else { 0u8 };
                
                genotypes.push(Some(vec![left_allele, right_allele]));
            }
            
            variants.push(Variant {
                position,
                genotypes,
            });
        }
        
        // Construct data matrix
        let n_haplotypes = n_samples * 2;
        let mut data_matrix = Array2::<f64>::zeros((n_haplotypes, n_variants));
        let mut positions = Vec::with_capacity(n_variants);
        
        // Fill data matrix
        for (valid_idx, variant) in variants.iter().enumerate() {
            positions.push(variant.position);
            
            // Add each haplotype's data
            for (sample_idx, genotypes_opt) in variant.genotypes.iter().enumerate() {
                if let Some(genotypes) = genotypes_opt {
                    if genotypes.len() >= 2 {
                        // Left haplotype
                        let left_idx = sample_idx * 2;
                        data_matrix[[left_idx, valid_idx]] = genotypes[0] as f64;
                        
                        // Right haplotype
                        let right_idx = sample_idx * 2 + 1;
                        data_matrix[[right_idx, valid_idx]] = genotypes[1] as f64;
                    }
                }
            }
        }
        
        // Print matrix stats
        let (rows, cols) = data_matrix.dim();
        println!("Data matrix: {} rows (haplotypes) x {} columns (variants)", rows, cols);
        
        // Check data matrix properties
        let zeros_count = data_matrix.iter().filter(|&&x| x == 0.0).count();
        let ones_count = data_matrix.iter().filter(|&&x| x == 1.0).count();
        println!("Matrix contains {} zeros and {} ones", zeros_count, ones_count);
        
        // Check first few columns for variance
        for c in 0..5 {
            let col = data_matrix.slice(s![.., c]);
            let sum: f64 = col.iter().sum();
            let mean = sum / col.len() as f64;
            let sum_sq: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum();
            let var = sum_sq / col.len() as f64;
            println!("Column {} variance: {:.6}", c, var);
        }
        
        // Run PCA
        let mut pca = PCA::new();
        
        match pca.rfit(
            data_matrix.clone(), // data_matrix is consumed by rfit
            n_components,
            5, // oversampling parameter
            Some(42), // seed
            None, // no variance tolerance
        ) {
            Ok(transformed) => { // rfit now returns the transformed principal components directly
                println!("PCA rfit computation successful, transformed PCs obtained.");
                
                // The `transformed` variable now holds the PC coordinates.
                // The subsequent explicit call to pca.transform for this data is no longer needed.
                
                // Check for NaN values
                let nan_count_unfiltered = transformed.iter().filter(|&&x| x.is_nan()).count();
                let total_values = transformed.nrows() * transformed.ncols();
                
                println!("NaN check: {}/{} values are NaN ({:.2}%)", 
                         nan_count_unfiltered, total_values, 
                         100.0 * nan_count_unfiltered as f64 / total_values as f64);
                
                // Print first few PC values for inspection
                println!("First 3 rows of PC values:");
                // we don't panic if less than 3 rows or less than n_components are available
                for i in 0..std::cmp::min(3, transformed.nrows()) {
                    print!("Row {}: ", i);
                    for j in 0..std::cmp::min(n_components, transformed.ncols()) {
                        print!("{:.6} ", transformed[[i, j]]);
                    }
                    println!();
                }

                
                // Try the process with filtering rare variants
                println!("\nNow testing with filtered rare variants:");
                let mut filtered_columns = Vec::new();
                
                // Identify columns with reasonable MAF
                for c in 0..cols {
                    let col = data_matrix.slice(s![.., c]);
                    let sum: f64 = col.iter().sum();
                    let freq = sum / col.len() as f64;
                    
                    // Keep only variants with MAF between 5% and 95%
                    if freq >= 0.05 && freq <= 0.95 {
                        filtered_columns.push(c);
                    }
                }
                
                println!("After filtering: {}/{} variants remain", 
                         filtered_columns.len(), cols);
                
                if !filtered_columns.is_empty() {
                    let mut filtered_matrix = Array2::<f64>::zeros((rows, filtered_columns.len()));
                    
                    // Copy selected columns to new matrix
                    for (new_c, &old_c) in filtered_columns.iter().enumerate() {
                        for r in 0..rows {
                            filtered_matrix[[r, new_c]] = data_matrix[[r, old_c]];
                        }
                    }
                    
                    // Run PCA on filtered data
                    let mut pca_filtered = PCA::new();
                    match pca_filtered.rfit(
                        filtered_matrix, // filtered_matrix is consumed by rfit
                        n_components,
                        5,
                        Some(42),
                        None,
                    ) {
                        Ok(transformed_filtered) => { // rfit now returns transformed PCs
                            // let transformed_filtered = pca_filtered.transform(filtered_matrix).unwrap(); // This is now redundant
                            let nan_count_filtered = transformed_filtered.iter().filter(|&&x| x.is_nan()).count();
                            println!("Filtered PCA NaN check: {}/{} values are NaN", 
                                    nan_count_filtered, transformed_filtered.len());
                            
                            // Print the first 3 rows of filtered PC values to demonstrate they are valid
                            println!("First 3 rows of FILTERED PC values:");
                                // we don't panic if less than 3 rows or less than n_components are available
                            for i in 0..std::cmp::min(3, transformed_filtered.nrows()) {
                                print!("Row {}: ", i);
                                for j in 0..std::cmp::min(n_components, transformed_filtered.ncols()) {
                                    print!("{:.6} ", transformed_filtered[[i, j]]);
                                }
                                println!();
                            }
                            
                            assert_eq!(nan_count_filtered, 0, "Filtered PCA produced NaN values");
                        },
                        Err(e) => {
                            panic!("Filtered PCA computation failed: {}", e);
                        }
                    }
                }

                // We might have NaN values in unfiltered data.
                // With extremely rare variants, covariance matrix could have an
                // issue where the ratio between largest and smallest eigenvalues becomes extremely large.
                // We divide by the square root of eigenvalues
                println!("Unfiltered PCA may produce NaN values: {} NaNs", {nan_count_unfiltered});
            },
            Err(e) => {
                panic!("PCA computation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_many_variants_haplotypes_binary() -> Result<(), Box<dyn std::error::Error>> {
        // Monitor memory usage before data generation
        let mut sys = System::new_all();
        sys.refresh_all();
        let pid = sysinfo::get_current_pid().expect("Unable to get current PID");
        let process_before = sys.process(pid).expect("Unable to get current process");
        let initial_mem = process_before.memory();
        println!("Initial memory usage (KB): {}", initial_mem);

        // Dimensions: 88 haplotypes (rows) x 100,000 variants (columns)
        let n_rows = 88;
        let n_cols = 100_000;

        // Generate random binary data (0 or 1) to simulate haplotype alleles
        // Each entry is either 0.0 or 1.0
        let start_data_gen = Instant::now();
        let mut data = Array2::<f64>::zeros((n_rows, n_cols));
        let mut rng = rand::thread_rng();
        for i in 0..n_rows {
            for j in 0..n_cols {
                let allele = rng.gen_range(0..=1);
                data[[i, j]] = allele as f64;
            }
        }
        let data_gen_duration = start_data_gen.elapsed();
        println!(
            "Binary data generation completed in {:.2?}",
            data_gen_duration
        );

        // Instantiate PCA
        let mut pca = PCA::new();

        // We will do randomized SVD with e.g. 10 principal components
        // Oversampling to 5, fixed random seed
        let n_components = 10;
        let n_oversamples = 5;
        let seed = Some(42_u64);

        // Fit PCA with rfit and get transformed data
        let start_pca_rfit = Instant::now();
        // rfit consumes the input data matrix.
        let transformed = pca.rfit(data, n_components, n_oversamples, seed, None)?;
        let pca_rfit_duration = start_pca_rfit.elapsed();
        println!("Randomized PCA rfit (which now includes transformation) completed in {:.2?}", pca_rfit_duration);

        // Basic dimensional checks
        assert_eq!(
            transformed.nrows(),
            n_rows,
            "Row count of transformed data is incorrect"
        );
        assert_eq!(
            transformed.ncols(),
            n_components,
            "Column count of transformed data is incorrect"
        );

        // Verify that none of the values are NaN or infinite
        for row in 0..n_rows {
            for col in 0..n_components {
                let val = transformed[[row, col]];
                assert!(val.is_finite(), "PCA output contains non-finite value");
            }
        }

        // Check memory usage afterwards
        sys.refresh_all();
        let process_after = sys
            .process(pid)
            .expect("Unable to get current process after test");
        let final_mem = process_after.memory();
        println!("Final memory usage (KB): {}", final_mem);
        if final_mem < initial_mem {
            println!(
                "Note: process-reported memory decreased; this can happen if allocations were freed."
            );
        }

        println!("Test completed successfully with 100,000 variants x 88 haplotypes (binary).");
        Ok(())
    }

    #[test]
    fn test_large_genotype_matrix_pc_correlation() -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "\n[Controlled Structure Test] Generating matrix with exactly 3 real components..."
        );

        // Use same dimensions as the original genotype test
        let n_samples = 88;
        let n_variants = 10000;
        let n_real_components = 3; // Exactly 3 components represent true structure
        let signal_strength = [50.0, 20.0, 10.0]; // Stronger to weaker signals for each component

        // Set random seed for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Create orthogonal basis vectors for the signal components (QR decomposition)
        let random_basis = Array2::<f64>::from_shape_fn((n_samples, n_real_components), |_| {
            rng.gen_range(-1.0..1.0)
        });
        let (q, _) = random_basis.qr().unwrap();

        // Make sure we have orthogonal unit vectors for true factors
        let true_factors = q.slice(s![.., 0..n_real_components]).to_owned();

        // Create random loadings for each variant (n_real_components x n_variants)
        let mut loadings = Array2::<f64>::zeros((n_real_components, n_variants));
        for i in 0..n_real_components {
            for j in 0..n_variants {
                loadings[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        // Create the pure signal matrix by combining factors and loadings, with decreasing strengths
        let mut pure_signal = Array2::<f64>::zeros((n_samples, n_variants));
        for k in 0..n_real_components {
            let factor = true_factors.column(k);
            let loading = loadings.row(k);
            for i in 0..n_samples {
                for j in 0..n_variants {
                    pure_signal[[i, j]] += factor[i] * loading[j] * signal_strength[k];
                }
            }
        }

        // Add pure random noise
        let mut noise = Array2::<f64>::zeros((n_samples, n_variants));
        for i in 0..n_samples {
            for j in 0..n_variants {
                noise[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        // Final matrix = signal + noise
        let data = &pure_signal + &noise;

        println!(
            "[Controlled Test] Created data with {} real components and pure noise",
            n_real_components
        );
        println!("[Controlled Test] Matrix shape: {:?}", data.shape());

        // Fit method
        let n_components = 5;
        let mut rust_pca_fit = PCA::new();
        rust_pca_fit.fit(data.clone(), None)?;
        let rust_transformed_fit = rust_pca_fit.transform(data.clone())?;

        // Run Python PCA for comparison
        let file = NamedTempFile::new().unwrap();
        {
            let mut handle = file.as_file();
            for row in data.rows() {
                let line = row
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(handle, "{}", line).unwrap();
            }
        }

        let cmd_output = Command::new("python3")
            .args(&vec![
                "tests/pca.py",
                "--data_csv",
                file.path().to_str().unwrap(),
                "--n_components",
                &format!("{}", n_components),
            ])
            .output()
            .expect("Failed to run Python PCA");

        assert!(cmd_output.status.success(), "Python PCA process failed");

        let stdout_text = String::from_utf8_lossy(&cmd_output.stdout).to_string();
        let python_transformed = parse_transformed_csv_from_python(&stdout_text);

        // Get eigenvalues from the data for reference
        let mut centered_data = data.clone();
        let mean = centered_data.mean_axis(Axis(0)).unwrap();
        for i in 0..n_samples {
            for j in 0..n_variants {
                centered_data[[i, j]] -= mean[j];
            }
        }
        let cov_matrix = centered_data.dot(&centered_data.t()) / (n_samples as f64 - 1.0);
        let (mut eigenvalues, _) = cov_matrix.eigh(UPLO::Upper).unwrap();
        eigenvalues
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| b.partial_cmp(a).unwrap());
        let total_variance: f64 = eigenvalues.iter().take(n_components).sum();

        println!("\n[Comparison] Explained Variance:");
        println!("Component | Rust Eigenvalue |  %   | Status");
        println!("---------+----------------+------+--------");
        for i in 0..n_components {
            let variance_pct = (eigenvalues[i] / total_variance) * 100.0;
            let status = if i < n_real_components {
                "REAL SIGNAL"
            } else {
                "PURE NOISE"
            };
            println!(
                "    PC{:<2}  | {:>14.2} | {:>4.1}% | {}",
                i + 1,
                eigenvalues[i],
                variance_pct,
                status
            );
        }

        println!("\nTotal variance: {:.2}", total_variance);
        println!(
            "First {} PCs capture real structure, remaining PCs are pure noise",
            n_real_components
        );

        println!("\nCorrelations for fit method:");
        println!("Component | Correlation | Required | Status");
        println!("---------+------------+----------+--------");

        let mut all_real_components_match_fit = true;
        for pc_idx in 0..n_components {
            let rust_pc = rust_transformed_fit.column(pc_idx);
            let python_pc = python_transformed.column(pc_idx);
            let correlation = calculate_pearson_correlation(rust_pc, python_pc);
            let abs_correlation = correlation.abs();

            if pc_idx < n_real_components {
                let threshold = 0.95;
                if abs_correlation < threshold {
                    all_real_components_match_fit = false;
                    println!(
                        "    PC{:<2}  | {:>10.4} | >={:.2}     | ✗ FAILED",
                        pc_idx + 1,
                        abs_correlation,
                        threshold
                    );
                } else {
                    println!(
                        "    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ PASSED",
                        pc_idx + 1,
                        abs_correlation,
                        threshold
                    );
                }
            } else {
                println!(
                    "    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ IGNORED",
                    pc_idx + 1,
                    abs_correlation,
                    0.0
                );
            }
        }

        assert!(
            all_real_components_match_fit,
            "Real signal components do not match for fit method"
        );

        // Now the rfit method
        let mut rust_pca_rfit = PCA::new();
        // rfit consumes data and returns the transformed PCs.
        // Since `data` is used for both fit and rfit tests, we must clone it for rfit.
        let rust_transformed_rfit = rust_pca_rfit.rfit(data.clone(), n_components, 5, Some(42_u64), None)?;

        println!("\nCorrelations for rfit method:");
        println!("Component | Correlation | Required | Status");
        println!("---------+------------+----------+--------");

        let mut all_real_components_match_rfit = true;
        for pc_idx in 0..n_components {
            let rust_pc = rust_transformed_rfit.column(pc_idx);
            let python_pc = python_transformed.column(pc_idx);
            let correlation = calculate_pearson_correlation(rust_pc, python_pc);
            let abs_correlation = correlation.abs();

            if pc_idx < n_real_components {
                let threshold = 0.95;
                if abs_correlation < threshold {
                    all_real_components_match_rfit = false;
                    println!(
                        "    PC{:<2}  | {:>10.4} | >={:.2}     | ✗ FAILED",
                        pc_idx + 1,
                        abs_correlation,
                        threshold
                    );
                } else {
                    println!(
                        "    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ PASSED",
                        pc_idx + 1,
                        abs_correlation,
                        threshold
                    );
                }
            } else {
                println!(
                    "    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ IGNORED",
                    pc_idx + 1,
                    abs_correlation,
                    0.0
                );
            }
        }

        assert!(
            all_real_components_match_rfit,
            "Real signal components do not match for rfit method"
        );

        println!("\n[Controlled Test] Both fit and rfit methods match correctly");
        Ok(())
    }

    fn calculate_pearson_correlation(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        // Get means
        let mean1 = v1.mean().unwrap();
        let mean2 = v2.mean().unwrap();

        // Calculate numerator (covariance)
        let mut numerator = 0.0;
        for (x1, x2) in v1.iter().zip(v2.iter()) {
            numerator += (x1 - mean1) * (x2 - mean2);
        }

        // Calculate denominators (std devs)
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        for &x1 in v1.iter() {
            var1 += (x1 - mean1).powi(2);
        }
        for &x2 in v2.iter() {
            var2 += (x2 - mean2).powi(2);
        }

        // Calculate correlation
        numerator / (var1.sqrt() * var2.sqrt())
    }

    fn parse_transformed_csv_from_python(output_text: &str) -> Array2<f64> {
        let mut lines = Vec::new();
        let mut in_csv_block = false;
        for line in output_text.lines() {
            // Start capturing once we see "Transformed Data (CSV):"
            if line.starts_with("Transformed Data (CSV):") {
                in_csv_block = true;
                continue;
            }
            // Stop capturing if we see "Components (CSV):"
            if line.starts_with("Components (CSV):") {
                break;
            }
            // If we are in CSV block, gather lines that are presumably CSV
            if in_csv_block {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // For single-column outputs, there won't be commas but data is still valid
                lines.push(trimmed.to_string());
            }
        }

        if lines.is_empty() {
            return Array2::<f64>::zeros((0, 0));
        }

        let col_count = lines[0].split(',').count();
        let row_count = lines.len();

        let mut arr = Array2::<f64>::zeros((row_count, col_count));
        for (i, l) in lines.iter().enumerate() {
            let nums: Vec<f64> = l
                .split(',')
                .map(|x| {
                    let trimmed_val = x.trim();
                    match trimmed_val.parse::<f64>() {
                        Ok(val) => val,
                        Err(_) => 0.0,
                    }
                })
                .collect();

            for (j, &val) in nums.iter().enumerate() {
                arr[[i, j]] = val;
            }
        }

        arr
    }
}

#[cfg(test)]
mod model_persistence_tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use tempfile::NamedTempFile;
    use std::error::Error;
    use std::f64; // For f64::NAN

    const COMPARISON_TOLERANCE: f64 = 1e-12; // Tolerance for float comparisons

    // Helper function to compare Option<Array1<f64>>
    fn assert_optional_array1_equals(
        arr1_opt: Option<&Array1<f64>>,
        arr2_opt: Option<&Array1<f64>>,
        context_msg: &str,
    ) {
        match (arr1_opt, arr2_opt) {
            (Some(a1), Some(a2)) => {
                assert_eq!(a1.dim(), a2.dim(), "Dimension mismatch for {} Array1", context_msg);
                for (i, (v1, v2)) in a1.iter().zip(a2.iter()).enumerate() {
                    assert!(
                        (v1 - v2).abs() < COMPARISON_TOLERANCE,
                        "Value mismatch at index {} for {}: {} vs {}",
                        i, context_msg, v1, v2
                    );
                }
            }
            (None, None) => { /* Both are None, which is considered equal in this context */ }
            _ => panic!(
                "Optional Array1 mismatch for {}: one is Some, other is None. Arr1: {:?}, Arr2: {:?}",
                context_msg, arr1_opt.is_some(), arr2_opt.is_some()
            ),
        }
    }

    // Helper function to compare Option<Array2<f64>>
    fn assert_optional_array2_equals(
        arr1_opt: Option<&Array2<f64>>,
        arr2_opt: Option<&Array2<f64>>,
        context_msg: &str,
    ) {
        match (arr1_opt, arr2_opt) {
            (Some(a1), Some(a2)) => {
                assert_eq!(a1.dim(), a2.dim(), "Dimension mismatch for {} Array2", context_msg);
                for (idx, (v1, v2)) in a1.iter().zip(a2.iter()).enumerate() {
                    assert!(
                        (v1 - v2).abs() < COMPARISON_TOLERANCE,
                        "Value mismatch at flat index {} for {}: {} vs {}",
                        idx, context_msg, v1, v2
                    );
                }
            }
            (None, None) => { /* Both are None, considered equal */ }
            _ => panic!(
                "Optional Array2 mismatch for {}: one is Some, other is None. Arr1: {:?}, Arr2: {:?}",
                context_msg, arr1_opt.is_some(), arr2_opt.is_some()
            ),
        }
    }

    #[test]
    fn test_save_load_after_exact_fit() -> Result<(), Box<dyn Error>> {
        println!("--- Test: Save/Load after Exact Fit ---");
        let data = array![
            [1.0, 2.0, 3.0, 4.5, 5.0],
            [5.0, 6.0, 7.0, 8.5, 9.0],
            [9.0, 10.0, 11.0, 12.5, 13.0],
            [13.0, 14.0, 15.0, 16.5, 17.0],
            [17.0, 18.0, 19.0, 20.5, 21.0]
        ];
        let n_samples = data.nrows();
        let n_features = data.ncols();
        // When tolerance is None, fit should determine the max possible components
        let expected_components = std::cmp::min(n_samples, n_features);

        let mut pca_original = PCA::new();
        pca_original.fit(data.clone(), None)?; // Fit with exact PCA

        // Pre-save assertions
        assert!(pca_original.rotation.is_some(), "Original (exact) model rotation should be Some");
        assert!(pca_original.mean.is_some(), "Original (exact) model mean should be Some");
        assert!(pca_original.scale.is_some(), "Original (exact) model scale should be Some");
        assert_eq!(pca_original.rotation.as_ref().unwrap().ncols(), expected_components, "Original (exact) model component count mismatch");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();

        pca_original.save_model(file_path)?;
        println!("Exact model saved to: {:?}", file_path);

        let pca_loaded = PCA::load_model(file_path)?;
        println!("Exact model loaded successfully.");

        // 1. Verify loaded model parameters are identical
        assert_optional_array2_equals(pca_original.rotation.as_ref(), pca_loaded.rotation.as_ref(), "rotation matrix (exact fit)");
        assert_optional_array1_equals(pca_original.mean.as_ref(), pca_loaded.mean.as_ref(), "mean vector (exact fit)");
        assert_optional_array1_equals(pca_original.scale.as_ref(), pca_loaded.scale.as_ref(), "scale vector (exact fit)");

        // 2. Verify transformation results are identical
        let data_to_transform = array![
            [2.0, 3.0, 4.0, 5.5, 6.0],
            [6.0, 7.0, 8.0, 9.5, 10.0]
        ];
        let transformed_original = pca_original.transform(data_to_transform.clone())?;
        let transformed_loaded = pca_loaded.transform(data_to_transform.clone())?;

        assert_eq!(transformed_original.dim(), transformed_loaded.dim(), "Transformed data dimension mismatch (exact fit)");
        for (val_orig, val_load) in transformed_original.iter().zip(transformed_loaded.iter()) {
            assert!((val_orig - val_load).abs() < COMPARISON_TOLERANCE, "Mismatch in transformed data after load (exact fit): {} vs {}", val_orig, val_load);
        }
        println!("Save/Load test for exact fit passed.");
        Ok(())
    }

    #[test]
    fn test_save_load_after_randomized_fit() -> Result<(), Box<dyn Error>> {
        println!("--- Test: Save/Load after Randomized Fit ---");
        let data = array![
            [1.5, 2.5, 3.5, 4.0, 0.5],
            [5.5, 6.5, 7.5, 8.0, 1.0],
            [9.5, 10.5, 11.5, 12.0, 1.5],
            [13.5, 14.5, 15.5, 16.0, 2.0],
            [17.5, 18.5, 19.5, 20.0, 2.5]
        ];
        let n_components_to_fit = 3;

        let mut pca_original = PCA::new();
        // rfit now returns the transformed PCs, but we only need to fit the model here.
        // The data is cloned as it's used to populate pca_original.
        let _ = pca_original.rfit(data.clone(), n_components_to_fit, 10, Some(123), None)?; // Using more oversamples

        // Pre-save assertions
        assert!(pca_original.rotation.is_some(), "Original (randomized) model rotation should be Some");
        assert_eq!(pca_original.rotation.as_ref().unwrap().ncols(), n_components_to_fit, "Original (randomized) model component count mismatch");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();
        pca_original.save_model(file_path)?;
        println!("Randomized model saved to: {:?}", file_path);

        let pca_loaded = PCA::load_model(file_path)?;
        println!("Randomized model loaded successfully.");

        // 1. Verify loaded model parameters
        assert_optional_array2_equals(pca_original.rotation.as_ref(), pca_loaded.rotation.as_ref(), "rotation matrix (randomized fit)");
        assert_optional_array1_equals(pca_original.mean.as_ref(), pca_loaded.mean.as_ref(), "mean vector (randomized fit)");
        assert_optional_array1_equals(pca_original.scale.as_ref(), pca_loaded.scale.as_ref(), "scale vector (randomized fit)");

        // 2. Verify transformation results
        let data_to_transform = array![
            [2.5, 3.5, 4.5, 5.0, 0.0],
            [6.5, 7.5, 8.5, 9.0, 1.0]
        ];
        let transformed_original = pca_original.transform(data_to_transform.clone())?;
        let transformed_loaded = pca_loaded.transform(data_to_transform.clone())?;

        assert_eq!(transformed_original.dim(), transformed_loaded.dim(), "Transformed data dimension mismatch (randomized fit)");
        for (val_orig, val_load) in transformed_original.iter().zip(transformed_loaded.iter()) {
            assert!((val_orig - val_load).abs() < COMPARISON_TOLERANCE, "Mismatch in transformed data after load (randomized fit): {} vs {}", val_orig, val_load);
        }
        println!("Save/Load test for randomized fit passed.");
        Ok(())
    }

    #[test]
    fn test_save_load_model_with_zero_components() -> Result<(), Box<dyn Error>> {
        println!("--- Test: Save/Load Model with Zero Components ---");
        let data = array![ // Data that will likely result in 0 components with high tolerance
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];

        let mut pca_original = PCA::new();
        // Use a very high tolerance that should mean no components are kept
        pca_original.fit(data.clone(), Some(0.999999999))?; 

        assert!(pca_original.rotation.is_some(), "Original model (zero components) rotation should be Some");
        assert_eq!(pca_original.rotation.as_ref().unwrap().ncols(), 0, "Model with no significant variance should have 0 components");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();
        pca_original.save_model(file_path)?;
        println!("Zero-component model saved to: {:?}", file_path);

        let pca_loaded = PCA::load_model(file_path)?;
        println!("Zero-component model loaded successfully.");

        assert_optional_array2_equals(pca_original.rotation.as_ref(), pca_loaded.rotation.as_ref(), "rotation (zero components)");
        assert_eq!(pca_loaded.rotation.as_ref().unwrap().ncols(), 0, "Loaded model should have 0 components");

        let data_to_transform = array![[1.1, 1.1, 1.1], [2.2, 2.2, 2.2]];
        let transformed_original = pca_original.transform(data_to_transform.clone())?;
        let transformed_loaded = pca_loaded.transform(data_to_transform.clone())?;
        
        assert_eq!(transformed_original.ncols(), 0, "Original transform output should have 0 columns");
        assert_eq!(transformed_loaded.ncols(), 0, "Loaded transform output should have 0 columns");
        assert_eq!(transformed_original.nrows(), data_to_transform.nrows());
        assert_eq!(transformed_loaded.nrows(), data_to_transform.nrows());
        
        println!("Save/Load test for zero-component model passed.");
        Ok(())
    }

    #[test]
    fn test_with_model_constructor_and_persistence() -> Result<(), Box<dyn Error>> {
        println!("--- Test: `with_model` Constructor and Persistence ---");
        let d_features = 4;
        let k_components = 2;
        let rotation = Array2::from_shape_vec((d_features, k_components), vec![
            0.5, 0.5, 
            -0.5, 0.5, 
            0.5, -0.5,
            -0.5, -0.5
        ])?; // Example orthonormal columns
        let mean = Array1::from(vec![10.0, 20.0, 30.0, 40.0]);
        let raw_std_devs = Array1::from(vec![1.0, 0.0000000001, 2.0, 0.0]); // One near-zero, one zero

        let pca_original = PCA::with_model(rotation.clone(), mean.clone(), raw_std_devs.clone())?;

        // Verify internal sanitization of scale
        let expected_sanitized_scale = array![1.0, 1.0, 2.0, 1.0];
        assert_optional_array1_equals(Some(&expected_sanitized_scale), pca_original.scale.as_ref(), "sanitized scale in with_model");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();
        pca_original.save_model(file_path)?;
        println!("Model created with `with_model` saved to: {:?}", file_path);
        
        let pca_loaded = PCA::load_model(file_path)?;
        println!("Model loaded successfully.");

        assert_optional_array2_equals(pca_original.rotation.as_ref(), pca_loaded.rotation.as_ref(), "rotation (with_model)");
        assert_optional_array1_equals(pca_original.mean.as_ref(), pca_loaded.mean.as_ref(), "mean (with_model)");
        assert_optional_array1_equals(pca_original.scale.as_ref(), pca_loaded.scale.as_ref(), "scale (with_model, should be sanitized)");
        assert_optional_array1_equals(Some(&expected_sanitized_scale), pca_loaded.scale.as_ref(), "loaded scale should match expected sanitized");


        let data_to_transform = array![
            [11.0, 20.0, 32.0, 40.0], // Orig: [1,0,2,0], Centered: [1,0,2,0], Scaled by [1,1,2,1]: [1,0,1,0]
            [10.0, 21.0, 30.0, 41.0]  // Orig: [0,1,0,1], Centered: [0,1,0,1], Scaled by [1,1,2,1]: [0,1,0,1]
        ];
        // Expected projection for P1 ([1,0,1,0]):
        // PC1: 1*0.5 + 0*(-0.5) + 1*0.5 + 0*(-0.5) = 0.5 + 0.5 = 1.0
        // PC2: 1*0.5 + 0*0.5    + 1*(-0.5)+ 0*(-0.5) = 0.5 - 0.5 = 0.0
        // Expected projection for P2 ([0,1,0,1]):
        // PC1: 0*0.5 + 1*(-0.5) + 0*0.5 + 1*(-0.5) = -0.5 - 0.5 = -1.0
        // PC2: 0*0.5 + 1*0.5    + 0*(-0.5)+ 1*(-0.5) =  0.5 - 0.5 =  0.0
        let expected_transformed = array![[1.0, 0.0], [-1.0, 0.0]];
        
        let transformed_loaded = pca_loaded.transform(data_to_transform)?;
        
        assert_eq!(transformed_loaded.dim(), (2, k_components), "Transformed data dimension mismatch (with_model)");
        for r in 0..transformed_loaded.nrows() {
            for c in 0..transformed_loaded.ncols() {
                assert!((transformed_loaded[[r,c]] - expected_transformed[[r,c]]).abs() < COMPARISON_TOLERANCE,
                    "Mismatch at [{},{}] (with_model): {} vs {}", r,c, transformed_loaded[[r,c]], expected_transformed[[r,c]]);
            }
        }
        println!("`with_model` constructor, persistence, and transform test passed.");
        Ok(())
    }

    #[test]
    fn test_load_model_error_conditions() -> Result<(), Box<dyn Error>> {
        println!("--- Test: `load_model` Error Conditions ---");
        let d_features = 3;
        let k_components = 2;
        let rotation_valid = Array2::zeros((d_features, k_components));
        let mean_valid = Array1::zeros(d_features);
        let scale_valid_sanitized = Array1::ones(d_features);

        // 1. Test loading non-existent file
        assert!(PCA::load_model("a_surely_non_existent_file.pca_model").is_err(), "Loading non-existent file should fail");

        // 2. Test loading a file that is not a valid bincode PCA model
        let empty_temp_file = NamedTempFile::new()?;
        // File is empty, so deserialization should fail
        assert!(PCA::load_model(empty_temp_file.path()).is_err(), "Loading an empty file should fail deserialization");

        // 3. Test loading a model with inconsistent dimensions (crafted struct, then saved)
        let rotation_bad_dim = Array2::zeros((d_features + 1, k_components)); // Mismatched feature count
        let bad_dim_pca_struct = PCA {
            rotation: Some(rotation_bad_dim),
            mean: Some(mean_valid.clone()),
            scale: Some(scale_valid_sanitized.clone()),
            explained_variance: None,
        };
        let temp_file_bad_dim = NamedTempFile::new()?;
        bad_dim_pca_struct.save_model(temp_file_bad_dim.path())?;
        assert!(PCA::load_model(temp_file_bad_dim.path()).is_err(), "Load should fail for inconsistent dimensions in saved model");

        // 4. Test loading a model with invalid scale vector (e.g., containing zero after it should have been sanitized)
        // To test load_model's check, we construct a PCA struct with a non-sanitized scale (containing zero)
        // and save it. load_model should then detect the zero in the scale upon loading.
        let scale_with_zero = Array1::from_vec(vec![1.0, 0.0, 2.0]);
        let zero_scale_pca_struct = PCA {
            rotation: Some(rotation_valid.clone()),
            mean: Some(mean_valid.clone()),
            scale: Some(scale_with_zero), // This scale contains a zero
            explained_variance: None,
        };
        let temp_file_zero_scale = NamedTempFile::new()?;
        zero_scale_pca_struct.save_model(temp_file_zero_scale.path())?;
        assert!(PCA::load_model(temp_file_zero_scale.path()).is_err(), "Load should fail for scale vector with zero");
        
        println!("`load_model` error condition tests passed.");
        Ok(())
    }

    #[test]
    fn test_with_model_scale_sanitization() -> Result<(), Box<dyn Error>> {
        println!("--- Test: `with_model` Scale Sanitization ---");
        let d_features = 5;
        let k_components = 2;
        let rotation = Array2::zeros((d_features, k_components)); // Dummy rotation
        let mean = Array1::zeros(d_features); // Dummy mean
        
        // Test with various problematic finite values for raw_standard_deviations
        let raw_stds_problematic_finite = array![-5.0, 0.0, 1e-10, 2.0, 0.5];
        // Expected outcome: non-positive or very small values become 1.0, others are preserved.
        let expected_sanitized_finite = array![1.0, 1.0, 1.0, 2.0, 0.5];
        
        let pca_model_finite = PCA::with_model(rotation.clone(), mean.clone(), raw_stds_problematic_finite.clone())?;
        assert_optional_array1_equals(
            Some(&expected_sanitized_finite), 
            pca_model_finite.scale.as_ref(), 
            "scale sanitization for problematic finite values in with_model"
        );

        // Test that an error is returned for non-finite values by the initial check in `with_model`.
        let raw_stds_with_nan = array![1.0, f64::NAN, 2.0];
        assert!(
            PCA::with_model(rotation.clone(), mean.clone(), raw_stds_with_nan).is_err(),
            "with_model should error on non-finite raw_standard_deviations due to the explicit check"
        );

        let raw_stds_with_inf = array![1.0, f64::INFINITY, 2.0];
         assert!(
            PCA::with_model(rotation.clone(), mean.clone(), raw_stds_with_inf).is_err(),
            "with_model should error on non-finite (infinity) raw_standard_deviations due to the explicit check"
        );

        println!("`with_model` scale sanitization tests passed.");
        Ok(())
    }

    #[test]
    fn test_load_model_rejects_non_positive_scale() -> Result<(), Box<dyn Error>> {
        println!("--- Test: `load_model` Rejects Non-Positive Scale ---");
        let d_features = 3;
        let k_components = 1;
        // Use valid rotation and mean for constructing test PCA structs.
        let rotation_valid = Array2::zeros((d_features, k_components));
        let mean_valid = Array1::zeros(d_features);

        // Case 1: Scale with a negative value.
        // Such a PCA struct might be created if `with_model` was different or from an external source.
        let scale_with_negative = array![-1.0, 1.0, 2.0];
        let pca_negative_scale = PCA {
            rotation: Some(rotation_valid.clone()),
            mean: Some(mean_valid.clone()),
            scale: Some(scale_with_negative),
            explained_variance: None,
        };
        let temp_file_neg_scale = NamedTempFile::new()?;
        // Save this struct which has a negative scale.
        // `save_model` itself doesn't validate the PCA internals, only that fields are Some.
        pca_negative_scale.save_model(temp_file_neg_scale.path())?; 
        // `load_model` should now reject this due to the `val <= 0.0` check.
        assert!(
            PCA::load_model(temp_file_neg_scale.path()).is_err(),
            "Load should fail for a saved model with a negative value in its scale vector."
        );
        
        // Case 2: Scale with zero (this scenario is also covered by test_load_model_error_conditions).
        // Re-confirming that the `val <= 0.0` check correctly handles this.
        let scale_with_zero = Array1::from_vec(vec![1.0, 0.0, 2.0]);
        let pca_zero_scale = PCA {
            rotation: Some(rotation_valid.clone()),
            mean: Some(mean_valid.clone()),
            scale: Some(scale_with_zero),
            explained_variance: None,
        };
        let temp_file_zero_scale = NamedTempFile::new()?;
        pca_zero_scale.save_model(temp_file_zero_scale.path())?;
         assert!(
            PCA::load_model(temp_file_zero_scale.path()).is_err(),
            "Load should fail for a saved model with a zero value in its scale vector."
        );

        println!("`load_model` rejection of non-positive scale tests passed.");
        Ok(())
    }
}

#[cfg(test)]
mod pca_tests {
    use std::io::Write;
    use std::process::Command;
    use tempfile::NamedTempFile;

    /// This function runs Python-based PCA as a reference and compares it to our Rust PCA.
    /// It writes the input data to a temporary CSV file, invokes `tests/pca.py` with the specified
    /// number of components (and, if `is_rpca` is true, uses rfit in Rust).
    /// It then parses the Python-transformed output, compares it to the Rust-transformed output,
    /// and fails the test if they differ beyond the given tolerance.
    pub fn run_python_pca_test(
        input: &ndarray::Array2<f64>,
        n_components: usize,
        is_rpca: bool,
        oversamples: usize,
        seed: Option<u64>,
        tol: f64,
        test_name: &str,
    ) {
        fn parse_transformed_csv_from_python(output_text: &str) -> ndarray::Array2<f64> {
            println!("[Rust Debug] Entering parse_transformed_csv_from_python...");
            let mut lines = Vec::new();
            let mut in_csv_block = false;
            for (line_idx, line) in output_text.lines().enumerate() {
                // println!("[Rust Debug] Raw line {}: {:?}", line_idx, line);
                // Start capturing once we see "Transformed Data (CSV):"
                if line.starts_with("Transformed Data (CSV):") {
                    println!("[Rust Debug] Found start of CSV block at line {}", line_idx);
                    in_csv_block = true;
                    continue;
                }
                // Stop capturing if we see "Components (CSV):"
                if line.starts_with("Components (CSV):") {
                    println!("[Rust Debug] Found end of CSV block at line {}", line_idx);
                    break;
                }
                // If we are in CSV block, gather lines that are presumably CSV
                if in_csv_block {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        println!("[Rust Debug] Skipping empty line in CSV block.");
                        continue;
                    }
                    // For single-column outputs, there won't be commas but data is still valid
                    println!("[Rust Debug] Accepting line: {:?}", trimmed);
                    lines.push(trimmed.to_string());
                }
            }

            println!("[Rust Debug] Collected {} CSV lines.", lines.len());
            if lines.is_empty() {
                println!("[Rust Debug] No CSV lines found. Returning empty array.");
                return ndarray::Array2::<f64>::zeros((0, 0));
            }
            let col_count = lines[0].split(',').count();
            let row_count = lines.len();
            println!(
                "[Rust Debug] Parsing into {} rows x {} cols.",
                row_count, col_count
            );

            let mut arr = ndarray::Array2::<f64>::zeros((row_count, col_count));
            for (i, l) in lines.iter().enumerate() {
                // println!("[Rust Debug] Parsing CSV row {}: {:?}", i, l);
                let nums: Vec<f64> = l
                    .split(',')
                    .map(|x| {
                        let trimmed_val = x.trim();
                        match trimmed_val.parse::<f64>() {
                            Ok(val) => {
                                // println!("[Rust Debug] Parsed float: {}", val);
                                val
                            }
                            Err(e) => {
                                println!("[Rust Debug] Parse error for {:?}: {:?}", trimmed_val, e);
                                0.0
                            }
                        }
                    })
                    .collect();

                for (j, &val) in nums.iter().enumerate() {
                    arr[[i, j]] = val;
                }
            }
            println!(
                "[Rust Debug] Final parsed array shape = ({}, {}).",
                arr.nrows(),
                arr.ncols()
            );
            arr
        }

        fn compare_pca_outputs_allow_sign_flip(
            rust_mat: &ndarray::Array2<f64>,
            py_mat: &ndarray::Array2<f64>,
            tol: f64,
        ) -> bool {
            if rust_mat.shape() != py_mat.shape() {
                return false;
            }
            let ncols = rust_mat.ncols();
            for c in 0..ncols {
                let col_r = rust_mat.column(c);
                let col_p = py_mat.column(c);
                let same_sign = col_r
                    .iter()
                    .zip(col_p.iter())
                    .all(|(&a, &b)| (a - b).abs() < tol);
                let opp_sign = col_r
                    .iter()
                    .zip(col_p.iter())
                    .all(|(&a, &b)| (a + b).abs() < tol);
                if !(same_sign || opp_sign) {
                    return false;
                }
            }
            true
        }

        let file = NamedTempFile::new().unwrap();
        {
            let mut handle = file.as_file();
            for row in input.rows() {
                let line = row
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(handle, "{}", line).unwrap();
            }
        }

        let comps_str = format!("{}", n_components);
        let args_vec = vec![
            "tests/pca.py",
            "--data_csv",
            file.path().to_str().unwrap(),
            "--n_components",
            &comps_str,
        ];

        let cmd_output = Command::new("python3")
            .args(&args_vec)
            .output()
            .expect(&format!("Failed to run python script for {}", test_name));

        assert!(
            cmd_output.status.success(),
            "Python script did not succeed in {}",
            test_name
        );

        let stdout_text = String::from_utf8_lossy(&cmd_output.stdout).to_string();
        let python_transformed = parse_transformed_csv_from_python(&stdout_text);

        let mut pca = super::PCA::new();
        let rust_transformed = if is_rpca {
            // rfit consumes the input and returns the transformed PCs.
            // Input is cloned as it was originally cloned for rfit then again for transform.
            pca.rfit(input.clone(), n_components, oversamples, seed, None).unwrap()
        } else {
            pca.fit(input.clone(), None).unwrap(); // fit does not return PCs
            // so, transform is still needed after fit.
            pca.transform(input.clone()).unwrap()
        };

        if !compare_pca_outputs_allow_sign_flip(&rust_transformed, &python_transformed, tol) {
            eprintln!("[Test: {}] => PCA mismatch with Python", test_name);
            eprintln!("EXPECTED (from Python):\n{:?}", python_transformed);
            eprintln!("ACTUAL   (from Rust):\n{:?}", rust_transformed);
            eprintln!("Rust PCA Rotation:\n{:?}", pca.rotation);
            panic!("Comparison with Python PCA failed in {}", test_name);
        }
    }

    use super::*;
    use ndarray::array;
    use ndarray_rand::rand_distr::Distribution;

    #[test]
    fn test_rpca_2x2() {
        let input = array![[0.5855288, -0.1093033], [0.7094660, -0.4534972]];
        run_python_pca_test(&input, 2, true, 0, Some(1926), 1e-6, "test_rpca_2x2");
    }

    #[test]
    fn test_rpca_2x2_k1() {
        let input = array![[0.5855288, -0.1093033], [0.7094660, -0.4534972]];
        run_python_pca_test(&input, 1, true, 0, Some(1926), 1e-6, "test_rpca_2x2_k1");
    }

    #[test]
    fn test_pca_2x2() {
        let input = array![[0.5855288, -0.1093033], [0.7094660, -0.4534972]];
        run_python_pca_test(&input, 2, false, 0, None, 1e-6, "test_pca_2x2");
    }

    #[test]
    fn test_pca_3x5() {
        let input = array![
            [0.5855288, -0.4534972, 0.6300986, -0.9193220, 0.3706279],
            [0.7094660, 0.6058875, -0.2761841, -0.1162478, 0.5202165],
            [-0.1093033, -1.8179560, -0.2841597, 1.8173120, -0.7505320]
        ];
        run_python_pca_test(&input, 5, false, 0, None, 1e-6, "test_pca_3x5");
    }

    #[test]
    fn test_pca_5x5() {
        let input = array![
            [0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219],
            [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851],
            [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284],
            [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374],
            [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095]
        ];
        run_python_pca_test(&input, 5, false, 0, None, 1e-6, "test_pca_5x5");
    }

    #[test]
    fn test_pca_5x7() {
        let input = array![
            [0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
            [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
            [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
            [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
            [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]
        ];
        run_python_pca_test(&input, 7, false, 0, None, 1e-6, "test_pca_5x7");
    }

    #[test]
    fn test_rpca_5x7_k4() {
        let input = array![
            [0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
            [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
            [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
            [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
            [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]
        ];
        run_python_pca_test(&input, 4, true, 0, Some(1926), 1e-6, "test_rpca_5x7_k4");
    }

    use ndarray::Array2;
    use ndarray_rand::RandomExt; // for creating random arrays
    use rand::distributions::Uniform;
    use rand::prelude::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // This helper function will make a random matrix that's size x size and check that there are no NaNs in the output
    fn test_pca_random(size: usize, seed: u64) {
        // Rng with input seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Input is a size x size matrix with elements randomly chosen from a uniform distribution between -1.0 and 1.0
        let input = Array2::<f64>::random_using((size, size), Uniform::new(-1.0, 1.0), &mut rng);

        // Transform the input with PCA
        let mut pca = PCA::new();
        pca.fit(input.clone(), None).unwrap();
        let output = pca.transform(input).unwrap();

        // Assert that none of the values in the output are NaN
        assert!(output.iter().all(|&x| !x.is_nan()));
    }

    #[test]
    fn test_pca_random_2() {
        test_pca_random(2, 1337);
    }

    #[test]
    fn test_pca_random_64() {
        test_pca_random(64, 1337);
    }

    #[test]
    fn test_pca_random_256() {
        test_pca_random(256, 1337);
    }

    #[test]
    fn test_pca_random_512() {
        test_pca_random(256, 1337);
    }

    fn test_pca_random_012(size: usize, seed: u64) {
        // Rng with input seed
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Input is a size x size matrix with elements randomly chosen from a uniform distribution between -1.0 and 1.0
        // n.b. we need to use a discrete distribution
        let input = Array2::<f64>::random_using(
            (size, size),
            Uniform::new_inclusive(0, 2).map(|x| x as f64),
            &mut rng,
        );

        // Transform the input with PCA
        let mut pca = PCA::new();
        pca.fit(input.clone(), None).unwrap();
        let output = pca.transform(input.clone()).unwrap();

        // Assert that none of the values in the output are NaN
        assert!(output.iter().all(|&x| !x.is_nan()));

        /*
        use std::fs::File;
        use std::io::Write;
        // write the input matrix to a file
        let mut file = File::create(format!("test_pca_random_012_{}_{}_input.csv", size, seed)).unwrap();
        input
            .rows()
            .into_iter()
            .for_each(|row| writeln!(file, "{}", row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",")).unwrap());

        // write the result to a file
        let mut file = File::create(format!("test_pca_random_012_{}_{}_output.csv", size, seed)).unwrap();
        output
            .rows()
            .into_iter()
        .for_each(|row| writeln!(file, "{}", row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",")).unwrap());
        */
    }

    #[test]
    fn test_pca_random_012_2() {
        test_pca_random_012(2, 1337);
    }

    #[test]
    fn test_pca_random_012_64() {
        test_pca_random_012(64, 1337);
    }

    #[test]
    fn test_pca_random_012_256() {
        test_pca_random_012(256, 1337);
    }

    #[test]
    fn test_pca_random_012_512() {
        test_pca_random_012(256, 1337);
    }

    /*
    #[test]
    fn test_pca_random_012_1024() {
        test_pca_random_012(1024, 1337);
    }*/

    #[test]
    #[should_panic]
    fn test_pca_fit_insufficient_samples() {
        let x = array![[1.0]];

        let mut pca = PCA::new();
        pca.fit(x, None).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_pca_transform_not_fitted() {
        let x = array![[1.0, 2.0]];

        let pca = PCA::new();
        let _ = pca.transform(x).unwrap();
    }
}


#[cfg(test)]
mod pca_bench_tests {
    use super::*;
    use ndarray::Array2;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;
    use sysinfo::System;

    /// Holds results for a single benchmark scenario.
    struct BenchResult {
        scenario_name: String,
        n_samples: usize,
        n_features: usize,
        fit_time: f64,
        fit_memory_kb: u64,
        rfit_time: f64,
        rfit_memory_kb: u64,
    }

    /// Generates random data of shape (n_samples x n_features) in [0, 1), seeded for reproducibility.
    fn generate_random_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(0.0..1.0))
    }

    /// Formats a memory size in KB into KB/MB/GB with 3 decimals where reasonable.
    fn format_memory_kb(mem_kb: u64) -> String {
        if mem_kb < 1024 {
            return format!("{} KB", mem_kb);
        }
        let mb = mem_kb as f64 / 1024.0;
        if mb < 1024.0 {
            return format!("{:.3} MB", mb);
        }
        let gb = mb / 1024.0;
        format!("{:.3} GB", gb)
    }

    /// Runs `fit` or `rfit` on the provided data, returns (time_secs, memory_kb) measured.
    fn benchmark_pca(
        use_rfit: bool,
        data: &Array2<f64>,
        n_components: usize,
        n_oversamples: usize,
        seed: u64,
    ) -> (f64, u64) {
        // Track memory usage before
        let mut sys = System::new_all();
        sys.refresh_all();
        let pid = sysinfo::get_current_pid().expect("Unable to get current PID");
        let process_start = sys.process(pid).expect("Unable to get current process");
        let initial_mem = process_start.memory();

        // Start timing
        let start_time = Instant::now();

        let mut pca = PCA::new();
        let transformed_data: Array2<f64>;

        if use_rfit {
            // rfit consumes data and returns transformed PCs.
            // The timing from start_time includes this combined operation.
            transformed_data = pca.rfit(data.clone(), n_components, n_oversamples, Some(seed), None)
                .expect("rfit failed");
        } else {
            pca.fit(data.clone(), None).expect("fit failed");
            // For 'fit', we still need to call 'transform' to get PCs.
            // The timing from start_time includes both fit and this transform.
            transformed_data = pca.transform(data.clone()).expect("transform failed");
        }

        // Assertions on the transformed data.
        assert_eq!(transformed_data.nrows(), data.nrows(), "Transformed data should have same number of rows as input.");
        let actual_components = pca.rotation().map_or(0, |r| r.ncols());
        assert_eq!(transformed_data.ncols(), actual_components, "Transformed data column count should match actual components in the model.");


        // Stop timing
        let duration = start_time.elapsed().as_secs_f64();

        // Track memory usage after
        sys.refresh_all();
        let process_end = sys
            .process(pid)
            .expect("Unable to get current process at end");
        let final_mem = process_end.memory();
        let used = if final_mem > initial_mem {
            final_mem - initial_mem
        } else {
            0
        };

        (duration, used)
    }

    /// Runs a single test scenario, returns the collected results.
    fn run_scenario(
        scenario_name: &str,
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> BenchResult {
        println!(
            "\n=== Test Case: {} ({} samples x {} features) ===",
            scenario_name, n_samples, n_features
        );

        let data = generate_random_data(n_samples, n_features, seed);
        let n_components = std::cmp::min(n_samples, n_features).min(30).max(2);
        let oversamples = 10;

        // FIT
        let (fit_time, fit_mem_kb) = benchmark_pca(false, &data, n_components, oversamples, seed);
        println!(
            "[fit]     Elapsed time: {:>7.3} s,  Memory usage: {}",
            fit_time,
            format_memory_kb(fit_mem_kb)
        );

        // RFIT
        let (rfit_time, rfit_mem_kb) = benchmark_pca(true, &data, n_components, oversamples, seed);
        println!(
            "[rfit]    Elapsed time: {:>7.3} s,  Memory usage: {}",
            rfit_time,
            format_memory_kb(rfit_mem_kb)
        );

        BenchResult {
            scenario_name: scenario_name.to_string(),
            n_samples,
            n_features,
            fit_time,
            fit_memory_kb: fit_mem_kb,
            rfit_time,
            rfit_memory_kb: rfit_mem_kb,
        }
    }

    /// Prints a final summary table of all scenario results.
    fn print_summary_table(results: &[BenchResult]) {
        println!("\n===== FINAL SUMMARY TABLE =====");
        println!(
            "{:<10} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Scenario", "Samples", "Features", "fit (s)", "fit Mem", "rfit (s)", "rfit Mem"
        );
        println!(
            "----------+----------+----------+------------+------------+------------+------------"
        );
        for r in results {
            println!(
                "{:<10} | {:>8} | {:>8} | {:>10.3} | {:>10} | {:>10.3} | {:>10}",
                r.scenario_name,
                r.n_samples,
                r.n_features,
                r.fit_time,
                format_memory_kb(r.fit_memory_kb),
                r.rfit_time,
                format_memory_kb(r.rfit_memory_kb)
            );
        }
        println!("================================");
    }

    /// Main test that runs all scenarios in sequence, printing immediate results,
    /// and ends with a summary table.
    #[test]
    fn bench_all_scenarios_in_one() {
        // Define our scenarios
        let scenarios = vec![
            ("Small", 100, 50, 1234),
            ("Medium", 1000, 500, 1234),
            ("Large", 5000, 2000, 1234),
            ("Square", 2000, 2000, 1234),
            ("Tall", 10000, 500, 1234),
            ("Wide", 500, 10000, 1234),
        ];

        let mut results = Vec::new();
        // Run them
        for (name, samps, feats, seed) in scenarios {
            let bench_result = run_scenario(name, samps, feats, seed);
            results.push(bench_result);
        }

        // Print final summary
        print_summary_table(&results);
    }
}
