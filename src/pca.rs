// Principal component analysis (PCA)

#![doc = include_str!("../README.md")]

use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, ArrayViewMut1, Axis, ShapeBuilder};
// UPLO is no longer needed as the backend's eigh_upper handles this.
// QR trait for .qr() and SVDInto for .svd_into() are replaced by backend calls.
// Eigh trait for .eigh() is replaced by backend calls.
use crate::linalg_backends::{BackendEigh, BackendQR, BackendSVD, LinAlgBackendProvider};
#[cfg(feature = "backend_faer")]
use faer::linalg::matmul::matmul;
#[cfg(feature = "backend_faer")]
use faer::{Accum, MatMut, MatRef, Par};
// use crate::ndarray_backend::NdarrayLinAlgBackend; // Replaced by LinAlgBackendProvider
// use crate::linalg_backend_dispatch::LinAlgBackendProvider; // Now part of linalg_backends
use rand::Rng;
use rand::SeedableRng; // For ChaCha8Rng::seed_from_u64
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal; // Distribution trait is implicitly used by Normal + rng.sample()

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// Helper function to calculate rank based on eigenvalue tolerance
fn calculate_rank_by_tolerance(
    sorted_desc_eigenvalues: &[f64],
    tolerance_fraction: Option<f64>,
    near_zero_threshold: f64, // Make sure this is passed, e.g., NEAR_ZERO_THRESHOLD
) -> usize {
    match tolerance_fraction {
        Some(tol_frac) => {
            let largest_eigval = sorted_desc_eigenvalues.first().copied().unwrap_or(0.0);

            if largest_eigval <= near_zero_threshold {
                return 0;
            }

            // Ensure tol_frac is clamped between 0.0 and 1.0
            let effective_tol_frac = tol_frac.max(0.0).min(1.0);
            let threshold_val = largest_eigval * effective_tol_frac;

            sorted_desc_eigenvalues
                .iter()
                .take_while(|&&val| val > threshold_val)
                .count()
        }
        None => sorted_desc_eigenvalues.len(),
    }
}

fn center_and_scale_columns(data_matrix: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n_samples = data_matrix.nrows();
    let n_features = data_matrix.ncols();

    const PARALLEL_COLUMN_THRESHOLD: usize = 256;

    let mut mean_vector = Array1::<f64>::zeros(n_features);
    let mut scale_vector = Array1::<f64>::zeros(n_features);

    let mean_slice = mean_vector
        .as_slice_mut()
        .expect("mean vector should be contiguous");
    let scale_slice = scale_vector
        .as_slice_mut()
        .expect("scale vector should be contiguous");

    #[inline]
    fn process_column(
        mut column: ArrayViewMut1<'_, f64>,
        mean_slot: &mut f64,
        scale_slot: &mut f64,
        n_samples: usize,
    ) {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &value in column.iter() {
            sum += value;
            sum_sq += value * value;
        }

        let n_samples_f64 = n_samples as f64;
        let mean = sum / n_samples_f64;
        let variance = if n_samples > 1 {
            let centered_sum_sq = (sum_sq - sum * sum / n_samples_f64).max(0.0);
            let var = centered_sum_sq / ((n_samples - 1) as f64);
            if var.is_finite() {
                var
            } else {
                0.0
            }
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        let sanitized_std = if !std_dev.is_finite() || std_dev <= NEAR_ZERO_THRESHOLD {
            1.0
        } else {
            std_dev
        };

        for value in column.iter_mut() {
            *value = (*value - mean) / sanitized_std;
        }

        *mean_slot = mean;
        *scale_slot = sanitized_std;
    }

    if n_features >= PARALLEL_COLUMN_THRESHOLD {
        data_matrix
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(mean_slice.par_iter_mut())
            .zip(scale_slice.par_iter_mut())
            .for_each(|((column, mean_slot), scale_slot)| {
                process_column(column, mean_slot, scale_slot, n_samples);
            });
    } else {
        data_matrix
            .axis_iter_mut(Axis(1))
            .into_iter()
            .zip(mean_slice.iter_mut())
            .zip(scale_slice.iter_mut())
            .for_each(|((column, mean_slot), scale_slot)| {
                process_column(column, mean_slot, scale_slot, n_samples);
            });
    }

    (mean_vector, scale_vector)
}

#[cfg(feature = "backend_faer")]
fn compute_covariance_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let (n_samples_total, n_features) = data_matrix.dim();
    if n_features == 0 {
        return Array2::zeros((0, 0));
    }
    let scale = 1.0 / ((n_samples - 1) as f64);
    let mut covariance = Array2::<f64>::zeros((n_features, n_features).f());

    let mut apply_matmul = |mat_a: MatRef<'_, f64>| {
        let cov_slice = covariance
            .as_slice_memory_order_mut()
            .expect("Covariance matrix should provide contiguous storage");
        let cov_dst = MatMut::from_column_major_slice_mut(cov_slice, n_features, n_features);
        matmul(
            cov_dst,
            Accum::Replace,
            mat_a.transpose(),
            mat_a,
            scale,
            Par::rayon(0),
        );
    };

    if let Some(slice) = data_matrix.as_slice_memory_order() {
        let mat_a = if data_matrix.is_standard_layout() {
            MatRef::from_row_major_slice(slice, n_samples_total, n_features)
        } else {
            MatRef::from_column_major_slice(slice, n_samples_total, n_features)
        };
        apply_matmul(mat_a);
    } else {
        let owned = data_matrix.to_owned();
        let slice = owned
            .as_slice_memory_order()
            .expect("Owned copy should be contiguous");
        let mat_a = MatRef::from_row_major_slice(slice, n_samples_total, n_features);
        apply_matmul(mat_a);
    }

    covariance
}

#[cfg(not(feature = "backend_faer"))]
fn compute_covariance_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let mut cov_matrix = data_matrix.t().dot(data_matrix);
    cov_matrix /= (n_samples - 1) as f64;
    cov_matrix
}

#[cfg(feature = "backend_faer")]
fn compute_gram_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let (n_samples_total, n_features) = data_matrix.dim();
    if n_samples_total == 0 {
        return Array2::zeros((0, 0));
    }
    let scale = 1.0 / ((n_samples - 1) as f64);
    let mut gram = Array2::<f64>::zeros((n_samples_total, n_samples_total).f());

    let mut apply_matmul = |mat_a: MatRef<'_, f64>| {
        let gram_slice = gram
            .as_slice_memory_order_mut()
            .expect("Gram matrix should provide contiguous storage");
        let gram_dst =
            MatMut::from_column_major_slice_mut(gram_slice, n_samples_total, n_samples_total);
        matmul(
            gram_dst,
            Accum::Replace,
            mat_a,
            mat_a.transpose(),
            scale,
            Par::rayon(0),
        );
    };

    if let Some(slice) = data_matrix.as_slice_memory_order() {
        let mat_a = if data_matrix.is_standard_layout() {
            MatRef::from_row_major_slice(slice, n_samples_total, n_features)
        } else {
            MatRef::from_column_major_slice(slice, n_samples_total, n_features)
        };
        apply_matmul(mat_a);
    } else {
        let owned = data_matrix.to_owned();
        let slice = owned
            .as_slice_memory_order()
            .expect("Owned copy should be contiguous");
        let mat_a = MatRef::from_row_major_slice(slice, n_samples_total, n_features);
        apply_matmul(mat_a);
    }

    gram
}

#[cfg(not(feature = "backend_faer"))]
fn compute_gram_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let mut gram_matrix = data_matrix.dot(&data_matrix.t());
    gram_matrix /= (n_samples - 1) as f64;
    gram_matrix
}

#[cfg(feature = "backend_faer")]
fn compute_feature_space_projection(
    data_matrix: &Array2<f64>,
    u_subset: &Array2<f64>,
) -> Array2<f64> {
    let (n_samples_total, n_features) = data_matrix.dim();
    let (_, final_rank) = u_subset.dim();

    if n_features == 0 || final_rank == 0 {
        return Array2::zeros((n_features, final_rank));
    }

    let mut rotation = Array2::<f64>::zeros((n_features, final_rank).f());
    let mut apply_matmul = |mat_a: MatRef<'_, f64>, mat_u: MatRef<'_, f64>| {
        let rot_slice = rotation
            .as_slice_memory_order_mut()
            .expect("Rotation matrix should provide contiguous storage");
        let rot_dst = MatMut::from_column_major_slice_mut(rot_slice, n_features, final_rank);
        matmul(
            rot_dst,
            Accum::Replace,
            mat_a.transpose(),
            mat_u,
            1.0,
            Par::rayon(0),
        );
    };

    let mut call_with_u = |mat_a: MatRef<'_, f64>| {
        if let Some(u_slice) = u_subset.as_slice_memory_order() {
            let mat_u = if u_subset.is_standard_layout() {
                MatRef::from_row_major_slice(u_slice, n_samples_total, final_rank)
            } else {
                MatRef::from_column_major_slice(u_slice, n_samples_total, final_rank)
            };
            apply_matmul(mat_a, mat_u);
        } else {
            let owned_u = u_subset.to_owned();
            let slice = owned_u
                .as_slice_memory_order()
                .expect("Owned copy should be contiguous");
            let mat_u = MatRef::from_row_major_slice(slice, n_samples_total, final_rank);
            apply_matmul(mat_a, mat_u);
        }
    };

    if let Some(a_slice) = data_matrix.as_slice_memory_order() {
        let mat_a = if data_matrix.is_standard_layout() {
            MatRef::from_row_major_slice(a_slice, n_samples_total, n_features)
        } else {
            MatRef::from_column_major_slice(a_slice, n_samples_total, n_features)
        };
        call_with_u(mat_a);
    } else {
        let owned_a = data_matrix.to_owned();
        let slice = owned_a
            .as_slice_memory_order()
            .expect("Owned copy should be contiguous");
        let mat_a = MatRef::from_row_major_slice(slice, n_samples_total, n_features);
        call_with_u(mat_a);
    }

    rotation
}

#[cfg(not(feature = "backend_faer"))]
fn compute_feature_space_projection(
    data_matrix: &Array2<f64>,
    u_subset: &Array2<f64>,
) -> Array2<f64> {
    data_matrix.t().dot(u_subset)
}

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
    pub rotation: Option<Array2<f64>>,
    /// Mean vector of the original training data.
    /// Shape: (n_features)
    pub mean: Option<Array1<f64>>,
    /// Sanitized scale vector, representing standard deviations of the original training data.
    /// This vector is guaranteed to contain only positive values.
    /// When set via `fit` or `rfit`, original standard deviations `s` where `s.abs() < 1e-9` are replaced by `1.0`.
    /// When set via `with_model`, input `raw_standard_deviations` `s` where `!s.is_finite()` or `s <= 1e-9` are replaced by `1.0`.
    /// Loaded models are also validated so scale factors are positive.
    /// Shape: (n_features)
    pub scale: Option<Array1<f64>>,
    /// Explained variance for each principal component (eigenvalues of the covariance matrix).
    /// Shape: (k_components)
    pub explained_variance: Option<Array1<f64>>,
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

// Public constants for thresholds and clamping values
pub const NEAR_ZERO_THRESHOLD: f64 = 1e-9;
pub const EIGENVALUE_CLAMP_MIN: f64 = 0.0;
pub const NORMALIZATION_THRESHOLD: f64 = 1e-9;
pub const SCALE_SANITIZATION_THRESHOLD: f64 = 1e-9;

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
            if !(d_features_rotation == 0
                && k_components == 0
                && d_features_mean == 0
                && d_features_raw_std == 0)
            {
                return Err(format!(
                    "PCA::with_model: Feature dimensions of rotation ({}), mean ({}), and raw_standard_deviations ({}) must match.",
                    d_features_rotation, d_features_mean, d_features_raw_std
                ).into());
            }
        }

        if d_features_rotation == 0 && k_components > 0 {
            return Err(
                "PCA::with_model: Rotation matrix has 0 features but expects components.".into(),
            );
        }

        if raw_standard_deviations.iter().any(|&val| !val.is_finite()) {
            // Explicitly reject non-finite inputs early.
            return Err("PCA::with_model: raw_standard_deviations contains non-finite (NaN or infinity) values.".into());
        }

        // Sanitize scale factors:
        // All scale factors are positive. Values that are not strictly positive (<= SCALE_SANITIZATION_THRESHOLD),
        // or were non-finite (though checked above), are replaced with 1.0.
        // This aligns `with_model`'s scale handling with `fit`/`rfit` and means `self.scale` is always positive.
        let sanitized_scale_vector = raw_standard_deviations.mapv(|val| {
            if val.is_finite() && val > SCALE_SANITIZATION_THRESHOLD {
                val
            } else {
                1.0
            }
        });

        Ok(Self {
            rotation: Some(rotation),
            mean: Some(mean),
            scale: Some(sanitized_scale_vector),
            explained_variance: None, // Explained variance is not provided by this constructor directly
        })
    }

    /// Returns a reference to the mean vector of the original training data, if computed.
    ///
    /// The mean vector has dimensions (n_features).
    /// Returns `None` if the PCA model has not been fitted.
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Returns a reference to the sanitized scale vector (standard deviations), if computed.
    ///
    /// The scale vector has dimensions (n_features) and contains positive values.
    /// Returns `None` if the PCA model has not been fitted.
    pub fn scale(&self) -> Option<&Array1<f64>> {
        self.scale.as_ref()
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
            return Err("PCA::fit: Input data_matrix has zero samples or zero features.".into());
        }
        if n_samples < 2 {
            return Err("PCA::fit: Input matrix must have at least 2 samples.".into());
        }

        let (mean_vector, sanitized_scale_vector) = center_and_scale_columns(&mut data_matrix);
        self.mean = Some(mean_vector);
        self.scale = Some(sanitized_scale_vector);

        let backend = LinAlgBackendProvider::<f64>::new();

        if n_features <= n_samples {
            let cov_matrix = compute_covariance_matrix(&data_matrix, n_samples);

            let eigh_result = backend.eigh_upper(&cov_matrix).map_err(|e| {
                format!(
                    "PCA::fit (Covariance path): Eigen decomposition of covariance matrix failed (via backend): {}",
                    e
                )
            })?;
            let eigenvalues = eigh_result.eigenvalues;
            let eigenvectors = eigh_result.eigenvectors;

            let eigenvalues_desc: Vec<f64> = eigenvalues.iter().rev().copied().collect();
            let rank_limit =
                calculate_rank_by_tolerance(&eigenvalues_desc, tolerance, NEAR_ZERO_THRESHOLD);

            let final_rank = std::cmp::min(rank_limit, n_features);

            if final_rank == 0 {
                self.rotation = Some(Array2::zeros((n_features, 0)));
                self.explained_variance = Some(Array1::zeros(0));
            } else {
                let mut explained_variance = Array1::<f64>::zeros(final_rank);
                let mut rotation_matrix = Array2::<f64>::zeros((n_features, final_rank));
                let total = eigenvalues.len();

                for component_idx in 0..final_rank {
                    let eigen_idx = total - 1 - component_idx;
                    let eigenvalue = eigenvalues[eigen_idx].max(EIGENVALUE_CLAMP_MIN);
                    explained_variance[component_idx] = eigenvalue;
                    rotation_matrix
                        .column_mut(component_idx)
                        .assign(&eigenvectors.column(eigen_idx));
                }

                self.rotation = Some(rotation_matrix);
                self.explained_variance = Some(explained_variance);
            }
        } else {
            let gram_matrix = compute_gram_matrix(&data_matrix, n_samples);

            let eigh_result_gram = backend.eigh_upper(&gram_matrix).map_err(|e| {
                format!(
                    "PCA::fit (Gram trick): Eigen decomposition of Gram matrix failed (via backend): {}",
                    e
                )
            })?;
            let gram_eigenvalues = eigh_result_gram.eigenvalues;
            let gram_eigenvectors_u = eigh_result_gram.eigenvectors;

            let eigenvalues_desc: Vec<f64> = gram_eigenvalues.iter().rev().copied().collect();
            let rank_limit =
                calculate_rank_by_tolerance(&eigenvalues_desc, tolerance, NEAR_ZERO_THRESHOLD);

            let final_rank = std::cmp::min(rank_limit, n_samples);

            if final_rank == 0 {
                self.rotation = Some(Array2::zeros((n_features, 0)));
                self.explained_variance = Some(Array1::zeros(0));
            } else {
                let mut explained_variance = Array1::<f64>::zeros(final_rank);
                let mut u_subset = Array2::<f64>::zeros((n_samples, final_rank));
                let total = gram_eigenvalues.len();

                for component_idx in 0..final_rank {
                    let eigen_idx = total - 1 - component_idx;
                    let eigenvalue = gram_eigenvalues[eigen_idx].max(EIGENVALUE_CLAMP_MIN);
                    explained_variance[component_idx] = eigenvalue;
                    u_subset
                        .column_mut(component_idx)
                        .assign(&gram_eigenvectors_u.column(eigen_idx));
                }

                let mut rotation_matrix = compute_feature_space_projection(&data_matrix, &u_subset);
                let scale_factors = explained_variance.map(|&lambda| {
                    let denom_squared = (n_samples - 1) as f64 * lambda;
                    if denom_squared > NEAR_ZERO_THRESHOLD * NEAR_ZERO_THRESHOLD {
                        1.0 / denom_squared.sqrt()
                    } else {
                        0.0
                    }
                });
                rotation_matrix *= &scale_factors;

                self.rotation = Some(rotation_matrix);
                self.explained_variance = Some(explained_variance);
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
    /// The core of this method is a randomized SVD algorithm (based on Halko, Martinsson, Tropp, 2011)
    /// that constructs a low-rank approximation of the input data. It adaptively chooses its
    /// sketching strategy based on the dimensions of the input data matrix `A`
    /// (n_samples × n_features, after centering and scaling):
    ///
    /// - If `n_features <= n_samples` (data matrix is tall or square, `D <= N`):
    ///   The algorithm directly sketches the input matrix `A` by forming `Y = A @ Omega'`, where `Omega'`
    ///   is a random Gaussian matrix of shape (n_features × l_sketch_components).
    ///   An orthonormal basis `Q_basis_prime` for the range of `Y` is found (N × l_sketch_components).
    ///   The data is then projected onto this basis: `B_projected_prime = Q_basis_prime.T @ A`
    ///   (l_sketch_components × n_features).
    ///
    /// - If `n_features > n_samples` (data matrix is wide, `D > N`):
    ///   To handle a large number of features efficiently, the algorithm sketches the
    ///   transpose `A.T`. It computes `Y = A.T @ Omega` (n_features × l_sketch_components),
    ///   where `Omega` is a random Gaussian matrix (n_samples × l_sketch_components).
    ///   An orthonormal basis `Q_basis` for the range of `Y` is found (n_features × l_sketch_components).
    ///   The data is then projected: `B_projected = Q_basis.T @ A.T = (A @ Q_basis).T`
    ///   (l_sketch_components × n_samples).
    ///
    /// In both cases, a few power iterations are used to refine the orthonormal basis (`Q_basis_prime` or `Q_basis`)
    /// for improved accuracy by better capturing the dominant singular vectors of `A`.
    ///
    /// An SVD is then performed on the smaller projected matrix (`B_projected_prime` or `B_projected`).
    /// The principal components (columns of the rotation matrix, stored in `self.rotation`)
    /// are derived from this SVD (from `V.T` in the `D <= N` case, or `Q_basis @ U` in the `D > N` case)
    /// and are normalized to unit length.
    /// The number of components kept can be influenced by the `tol` (tolerance) parameter,
    /// up to `n_components_requested`.
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
    /// * `n_oversamples` - Number of additional random dimensions (`p`) to sample during the sketching
    ///   phase, forming a sketch of size `l = k + p` (where `k` is `n_components_requested`).
    ///   This helps improve the accuracy of the randomized SVD.
    ///   - If `0`, an adaptive default for `p` is used (typically 10% of `n_components_requested`,
    ///     clamped between 5 and 20).
    ///   - If positive, this value is used for `p`, but an internal minimum (e.g., 4) is
    ///     enforced for robustness.
    ///   Recommended values when specifying explicitly: 5-20.
    /// * `seed` - Optional `u64` seed for the random number generator used in sketching, allowing
    ///   for reproducible results. If `None`, a random seed is used.
    /// * `tol` - Optional tolerance (a float between 0.0 and 1.0, exclusive of 0.0 if used for filtering).
    ///   If `Some(t_val)`, components are kept if their corresponding singular value `s_i` from the
    ///   internal SVD of the projected sketch satisfies `s_i > t_val * s_max`, where `s_max` is the
    ///   largest singular value from that SVD. The number of components kept will be at most
    ///   `n_components_requested`.
    ///   If `None`, tolerance-based filtering based on singular value ratios is skipped, and up to
    ///   `n_components_requested` components (or the effective rank of the sketch) are kept.
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
        n_oversamples: usize, // User's 'p' value or 0 for adaptive default
        seed: Option<u64>,
        tol: Option<f64>,
    ) -> Result<Array2<f64>, Box<dyn Error>> {
        let n_samples = x_input_data.nrows();
        let n_features = x_input_data.ncols();

        // --- 1. Input Validations ---
        if n_samples == 0 || n_features == 0 {
            return Err("PCA::rfit: Input matrix has zero samples or zero features.".into());
        }
        if n_samples < 2 {
            // PCA requires at least 2 samples to compute variance meaningfully.
            return Err("PCA::rfit: Input matrix must have at least 2 samples for PCA.".into());
        }
        if n_components_requested == 0 {
            return Err(
                "PCA::rfit: Number of requested components (n_components_requested) must be greater than 0."
                    .into(),
            );
        }

        // --- 2. Mean Centering and Scaling ---
        // This modifies x_input_data in place.
        let mean_vector = x_input_data
            .mean_axis(Axis(0))
            .ok_or("PCA::rfit: Failed to compute mean vector.")?;
        self.mean = Some(mean_vector.clone());
        x_input_data -= &mean_vector; // Center data

        let std_dev_vector_original = x_input_data.map_axis(Axis(0), |column| column.std(1.0));
        // Sanitize scale: replace near-zero std devs with 1.0 to avoid division by zero/instability.
        // Non-finite values (checked in `with_model`, but good practice for direct fit too if data could be raw)
        // are also mapped to 1.0. `std` should produce finite values from finite input.
        // SCALE_SANITIZATION_THRESHOLD is now a global const defined above
        let sanitized_scale_vector = std_dev_vector_original.mapv(|val| {
            if val.is_finite() && val.abs() > SCALE_SANITIZATION_THRESHOLD {
                val
            } else {
                1.0
            }
        });
        self.scale = Some(sanitized_scale_vector.clone());
        x_input_data /= &sanitized_scale_vector; // Scale data

        let centered_scaled_data_a = x_input_data; // N x D, alias for clarity

        // --- 3. Determine Target Components and Sketch Size (l = k + p) ---
        let max_possible_rank = std::cmp::min(n_samples, n_features);

        if max_possible_rank == 0 {
            // Should not happen if n_samples/n_features > 0, but defensive.
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
            return Ok(Array2::zeros((n_samples, 0)));
        }

        // Determine 'p' (oversampling amount) based on user input and robust defaults
        // Constants for adaptive oversampling heuristic based on Halko et al. recommendations
        const RFIT_ADAPTIVE_P_LOWER_BOUND: usize = 5; // Minimum adaptive oversampling
        const RFIT_ADAPTIVE_P_UPPER_BOUND: usize = 20; // Maximum adaptive oversampling
        const RFIT_MINIMUM_ROBUST_P_FLOOR: usize = 4; // Absolute minimum oversampling for theory

        let p_to_use: usize = if n_oversamples == 0 {
            // Sentinel for "use adaptive default"
            // Adaptive 'p' is ~10% of k, clamped.
            let p_adaptive_raw = (n_components_requested as f64 * 0.1).ceil() as usize;
            p_adaptive_raw.clamp(RFIT_ADAPTIVE_P_LOWER_BOUND, RFIT_ADAPTIVE_P_UPPER_BOUND)
        } else {
            // User provided a specific n_oversamples value
            n_oversamples.max(RFIT_MINIMUM_ROBUST_P_FLOOR) // So it's not critically low
        };

        let l_sketch_components_ideal = n_components_requested + p_to_use;

        // Clamp sketch components: cannot exceed dimensions of matrix, must be at least 1,
        // and should be at least n_components_requested (if possible within matrix dimensions).
        let mut l_sketch_components = l_sketch_components_ideal.min(max_possible_rank);
        l_sketch_components = l_sketch_components.max(1); // So at least 1 sketch component
                                                          // So sketch is large enough to find requested components, if data rank allows.
        l_sketch_components =
            l_sketch_components.max(n_components_requested.min(max_possible_rank));

        // If, after all clamping, sketch size is 0 (e.g. max_possible_rank was 0, though checked), handle gracefully.
        if l_sketch_components == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
            return Ok(Array2::zeros((n_samples, 0)));
        }

        // --- 4. Initialize RNG ---
        // Number of power iterations (q in Halko et al.) to improve accuracy. 2 is a common default.
        const N_POWER_ITERATIONS: usize = 1;
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(rand::thread_rng())
                .map_err(|e| format!("PCA::rfit: Failed to initialize RNG: {}", e))?,
        };

        let backend = LinAlgBackendProvider::<f64>::new(); // Use LinAlgBackendProvider

        // --- 5. Randomized SVD Core: Sketching and Projection ---
        // `final_rotation_sketch` will hold the unnormalized principal axes (D x k_eff_sketch)
        // `singular_values_from_projected_b` will hold singular values of the small projected matrix.
        let final_rotation_sketch: Array2<f64>;
        let singular_values_from_projected_b: Array1<f64>;

        if n_features <= n_samples {
            // --- Strategy 1: D <= N (Tall or Square Matrix) ---
            // Sketch A directly: Y = A @ Omega_prime (N x L)
            // Orthonormalize Y into Q_prime_basis (N x L)
            // Project: B_prime_projected = Q_prime_basis.T @ A (L x D)
            // SVD of B_prime_projected: U_B_prime @ S_B_prime @ VT_B_prime
            // Rotation sketch = VT_B_prime.T (D x L_eff)

            let omega_prime_shape = (n_features, l_sketch_components); // D x L
            let omega_prime_random_matrix = Array2::from_shape_fn(omega_prime_shape, |_| {
                rng.sample(Normal::new(0.0, 1.0).expect("Failed to create Normal distribution"))
            });

            // Initial sketch: Y = A @ Omega_prime
            let q_prime_basis_candidate = centered_scaled_data_a.dot(&omega_prime_random_matrix); // (N x D) @ (D x L) -> N x L
            if q_prime_basis_candidate.ncols() == 0 {
                // Should only happen if l_sketch_components was 0
                return Err("PCA::rfit (D <= N path): Sketch Q'_basis_candidate (from A) has zero columns before QR. This indicates l_sketch_components became 0.".into());
            }
            let mut q_prime_basis = backend.qr_q_factor(&q_prime_basis_candidate)
                .map_err(|e| format!("PCA::rfit (D <= N path): QR decomposition of Q' (initial sketch of A) failed: {}", e))?; // Q factor (N x L_actual)

            // Power iterations to refine Q_prime_basis
            for i in 0..N_POWER_ITERATIONS {
                if q_prime_basis.ncols() == 0 {
                    break;
                } // Orthonormal basis might have reduced rank
                  // W_prime_intermediate = A.T @ Q_prime_basis  (D x N) @ (N x L) -> D x L
                let w_prime_intermediate_candidate = centered_scaled_data_a.t().dot(&q_prime_basis);
                if w_prime_intermediate_candidate.ncols() == 0 {
                    break;
                }
                let w_prime_ortho_basis = backend.qr_q_factor(&w_prime_intermediate_candidate)
                    .map_err(|e| format!("PCA::rfit (D <= N path): QR decomposition of W' (power iteration {}) failed: {}", i, e))?; // D x L_actual

                if w_prime_ortho_basis.ncols() == 0 {
                    break;
                }
                // Z_prime_intermediate = A @ W_prime_ortho_basis (N x D) @ (D x L) -> N x L
                let z_prime_intermediate_candidate =
                    centered_scaled_data_a.dot(&w_prime_ortho_basis);
                if z_prime_intermediate_candidate.ncols() == 0 {
                    break;
                }
                q_prime_basis = backend.qr_q_factor(&z_prime_intermediate_candidate)
                    .map_err(|e| format!("PCA::rfit (D <= N path): QR decomposition of Z' (power iteration {}) failed: {}", i, e))?;
                // N x L_actual
            }

            if q_prime_basis.ncols() == 0 {
                return Err("PCA::rfit (D <= N path): Refined sketch Q'_basis (from A) has zero columns after power iterations.".into());
            }

            // Project A onto the refined basis: B_prime_projected = Q_prime_basis.T @ A
            let b_prime_projected = q_prime_basis.t().dot(&centered_scaled_data_a); // (L x N) @ (N x D) -> L x D

            // SVD of the small projected matrix B_prime_projected
            // We need V^T from B_prime = U S V^T.
            // The columns of V (rows of V^T) are the principal components in feature space.
            let svd_output_b_prime = backend
                .svd_into(b_prime_projected, false, true) // compute_u=false, compute_v=true
                .map_err(|e| {
                    format!(
                        "PCA::rfit (D <= N path): SVD of B' (projected sketch of A) failed: {}",
                        e
                    )
                })?;

            singular_values_from_projected_b = svd_output_b_prime.s; // s_values
            let vt_b_prime_matrix = svd_output_b_prime
                .vt // V^T matrix
                .ok_or(
                    "PCA::rfit (D <= N path): SVD V_B_prime^T not computed from B' (A sketch)",
                )?; // L_eff x D or L_actual x D

            final_rotation_sketch = vt_b_prime_matrix.t().into_owned(); // D x L_eff (Principal axes)
        } else {
            // --- Strategy 2: D > N (Wide Matrix: More features than samples) ---
            // Sketch A.T: Y = A.T @ Omega (D x L)
            // Orthonormalize Y into Q_basis (D x L)
            // Project: B_projected = Q_basis.T @ A.T = (A @ Q_basis).T (L x N)
            // SVD of B_projected: U_B @ S_B @ VT_B
            // Rotation sketch = Q_basis @ U_B (D x L_eff)

            let omega_shape = (n_samples, l_sketch_components); // N x L
            let omega_random_matrix = Array2::from_shape_fn(omega_shape, |_| {
                rng.sample(Normal::new(0.0, 1.0).expect("Failed to create Normal distribution"))
            });

            // Initial sketch: Y_aat_candidate = A.T @ Omega
            let q_basis_candidate = centered_scaled_data_a.t().dot(&omega_random_matrix); // (D x N) @ (N x L) -> D x L
            if q_basis_candidate.ncols() == 0 {
                return Err("PCA::rfit (D > N path): Initial sketch Q_basis_candidate (from A.T) has zero columns before QR. This indicates l_sketch_components became 0.".into());
            }
            let mut q_basis = backend.qr_q_factor(&q_basis_candidate)
                .map_err(|e| format!("PCA::rfit (D > N path): QR decomposition of Q (initial sketch of A.T) failed: {}",e))?; // Q factor (D x L_actual)

            // Power iterations to refine Q_basis
            for i in 0..N_POWER_ITERATIONS {
                if q_basis.ncols() == 0 {
                    break;
                }
                // W_intermediate_candidate = A @ Q_basis (N x D) @ (D x L) -> N x L
                let w_intermediate_candidate = centered_scaled_data_a.dot(&q_basis);
                if w_intermediate_candidate.ncols() == 0 {
                    break;
                }
                let w_ortho_basis = backend.qr_q_factor(&w_intermediate_candidate)
                    .map_err(|e| format!("PCA::rfit (D > N path): QR decomposition of W (power iteration {}) failed: {}", i, e))?; // N x L_actual

                if w_ortho_basis.ncols() == 0 {
                    break;
                }
                // Z_intermediate_candidate = A.T @ W_ortho_basis (D x N) @ (N x L) -> D x L
                let z_intermediate_candidate = centered_scaled_data_a.t().dot(&w_ortho_basis);
                if z_intermediate_candidate.ncols() == 0 {
                    break;
                }
                q_basis = backend.qr_q_factor(&z_intermediate_candidate)
                    .map_err(|e| format!("PCA::rfit (D > N path): QR decomposition of Z (power iteration {}) failed: {}", i, e))?;
                // D x L_actual
            }

            if q_basis.ncols() == 0 {
                return Err("PCA::rfit (D > N path): Refined sketch Q_basis (from A.T) has zero columns after power iterations.".into());
            }

            // Project A.T onto the refined basis Q_basis: B_projected = Q_basis.T @ A.T = (A @ Q_basis).T
            // So, B_projected_transpose = A @ Q_basis
            let b_projected_transpose = centered_scaled_data_a.dot(&q_basis); // (N x D) @ (D x L) -> N x L
            let b_projected = b_projected_transpose.t().into_owned(); // L x N (owned for SVD)

            // SVD of the small projected matrix B_projected
            // We need U_B from B_projected = U_B S_B VT_B.
            // Rotation sketch = Q_basis @ U_B
            let svd_output_b = backend
                .svd_into(b_projected, true, false) // compute_u=true, compute_v=false
                .map_err(|e| {
                    format!(
                        "PCA::rfit (D > N path): SVD of B_projected (from A.T sketch) failed: {}",
                        e
                    )
                })?;

            singular_values_from_projected_b = svd_output_b.s; // s_values
            let u_b_matrix = svd_output_b
                .u // U_B matrix
                .ok_or(
                    "PCA::rfit (D > N path): SVD U_B not computed from B_projected (A.T sketch)",
                )?; // L x L_eff or L_actual x L_eff

            final_rotation_sketch = q_basis.dot(&u_b_matrix); // (D x L_actual) @ (L_actual x L_eff) -> D x L_eff (Principal axes)
        }

        // --- 6. Post-Sketching: Select Components, Normalize, and Update Model ---

        // Number of components effectively found by the SVD of the sketch (can be <= l_sketch_components)
        let k_eff_from_sketch_svd = final_rotation_sketch.ncols();
        if k_eff_from_sketch_svd == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
            return Ok(Array2::zeros((n_samples, 0)));
        }

        // Determine the number of components to keep based on user request and tolerance
        let mut n_components_to_keep = n_components_requested.min(k_eff_from_sketch_svd);

        if let Some(tolerance_value) = tol {
            // Apply tolerance filtering only if valid and data allows
            if !singular_values_from_projected_b.is_empty()
                && tolerance_value > 0.0 // Tolerance must be positive
                && tolerance_value < 1.0
            // Tolerance strictly less than 1 makes sense for ratio filtering
            // Using <= 1.0 as in original code was too permissive,
            // as tol=1.0 would keep only components equal to the largest.
            {
                // SVD singular values are sorted in descending order.
                let largest_singular_value = singular_values_from_projected_b[0];
                if largest_singular_value > NEAR_ZERO_THRESHOLD {
                    // Avoid issues if all singular values are effectively zero
                    let singular_value_threshold = tolerance_value * largest_singular_value; // tolerance_value is a fraction, should be > 0
                    let rank_by_tolerance = singular_values_from_projected_b
                        .iter()
                        .take_while(|&&s_val| s_val > singular_value_threshold)
                        .count();
                    n_components_to_keep = n_components_to_keep.min(rank_by_tolerance);
                } else {
                    // All singular values are effectively zero, keep no components by tolerance.
                    n_components_to_keep = 0;
                }
            }
            // Note: If tolerance_value is outside (0,1), no filtering by tolerance is performed here.
            // User might intend tol=0 to keep all, or tol=1 to keep only max (if any are strictly > others).
            // The logic `tolerance_value < 1.0` makes it a relative threshold.
        }
        // n_components_to_keep is already non-negative due to min with k_eff_from_sketch_svd and rank_by_tolerance.

        if n_components_to_keep == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
        } else {
            // Slice the sketch to the number of components to keep before normalization
            let mut final_rotation_matrix = final_rotation_sketch
                .slice_axis(Axis(1), ndarray::Slice::from(0..n_components_to_keep)) // Corrected slice
                .to_owned(); // Shape: (n_features, n_components_to_keep)

            // Normalize the selected principal axes (columns) to unit length
            // NORMALIZATION_THRESHOLD is now a global const defined above
            for mut column_vec in final_rotation_matrix.columns_mut() {
                let norm_value = column_vec.dot(&column_vec).sqrt();
                if norm_value > NORMALIZATION_THRESHOLD {
                    column_vec.mapv_inplace(|val| val / norm_value);
                } else {
                    // If a component's sketch has a near-zero norm, it captured negligible variance
                    // or is numerically unstable. Set it to a zero vector.
                    column_vec.fill(EIGENVALUE_CLAMP_MIN);
                }
            }
            self.rotation = Some(final_rotation_matrix);

            // Calculate explained variance from the selected singular values
            if n_samples > 1 {
                let selected_singular_values =
                    singular_values_from_projected_b.slice(s![..n_components_to_keep]);
                // Variance = (singular_value_of_sketch)^2 / (n_samples - 1)
                let variances =
                    selected_singular_values.mapv(|s_val| s_val.powi(2) / ((n_samples - 1) as f64));
                self.explained_variance = Some(variances);
            } else {
                // Variance is undefined for a single sample.
                self.explained_variance = Some(Array1::from_elem(n_components_to_keep, f64::NAN));
            }
        }

        // --- 7. Calculate and Return Principal Component Scores for the Input Data ---
        // Scores = Centered_Scaled_Data_A @ Final_Rotation_Matrix
        // Need to access the potentially just-set self.rotation.
        let rotation_for_transform = self.rotation.as_ref().ok_or_else(|| {
            "PCA::rfit: Internal error: Rotation matrix not set after rfit processing."
        })?;

        // centered_scaled_data_a is (N x D)
        // rotation_for_transform is (D x k_kept)
        let transformed_principal_components = centered_scaled_data_a.dot(rotation_for_transform); // -> N x k_kept

        Ok(transformed_principal_components)
    }

    /// Applies the PCA transformation to the given data.
    ///
    /// The data is centered and scaled using the mean and scale factors
    /// learned during fitting (or loaded into the model), and then projected
    /// onto the principal components.
    ///
    /// * `x` - Input data to transform, shape (m_samples, d_features).
    ///         Can be a single sample (1 row) or multiple samples.
    ///         This matrix is modified in place.
    ///
    /// # Errors
    /// Returns an error if the PCA model is not fitted/loaded (i.e., missing mean,
    /// scale, or rotation components), or if the input data's feature dimension
    /// does not match the model's feature dimension.
    pub fn transform(&self, mut x: Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        // Retrieve model components, so they exist.
        let rotation_matrix = self.rotation.as_ref().ok_or_else(|| {
            "PCA::transform: PCA model: Rotation matrix not set. Fit or load a model first."
        })?;
        let mean_vector = self.mean.as_ref().ok_or_else(|| {
            "PCA::transform: PCA model: Mean vector not set. Fit or load a model first."
        })?;
        // self.scale is guaranteed to contain positive, finite values by model construction/loading.
        let scale_vector = self.scale.as_ref().ok_or_else(|| {
            "PCA::transform: PCA model: Scale vector not set. Fit or load a model first."
        })?;

        let n_input_samples = x.nrows();
        let n_input_features = x.ncols();
        let n_model_features = mean_vector.len(); // Also self.scale.len() and self.rotation.nrows()

        // Validate dimensions
        if n_input_features != n_model_features {
            return Err(format!(
                "PCA::transform: Input data feature dimension ({}) does not match model's feature dimension ({}).",
                n_input_features, n_model_features
            ).into());
        }
        // Additional internal consistency checks (should hold if model was properly constructed/loaded)
        // These checks are defensive programming.
        if rotation_matrix.nrows() != n_model_features {
            return Err(format!(
                "PCA::transform: Model inconsistency: Rotation matrix feature dimension ({}) does not match model's feature dimension ({}).",
                rotation_matrix.nrows(), n_model_features
            ).into());
        }
        if scale_vector.len() != n_model_features {
            return Err(format!(
                "PCA::transform: Model inconsistency: Scale vector dimension ({}) does not match model's feature dimension ({}).",
                scale_vector.len(), n_model_features
            ).into());
        }

        // Handle empty input data (0 samples)
        if n_input_samples == 0 {
            let k_components = rotation_matrix.ncols();
            return Ok(Array2::zeros((0, k_components))); // Return 0-sample matrix with correct number of components
        }

        // Fuse centering and scaling in a single pass over the data `x`.
        // This modifies `x` in place.
        // Iterate over each row of x (which is an ArrayViewMut1).
        for mut row in x.axis_iter_mut(Axis(0)) {
            // Zip::from iterates over the elements of the row, mean_vector, and scale_vector simultaneously.
            // `row.view_mut()` provides the necessary IntoNdProducer.
            // `mean_vector.view()` and `scale_vector.view()` also provide IntoNdProducer.
            ndarray::Zip::from(row.view_mut())
                .and(mean_vector.view())
                .and(scale_vector.view())
                .for_each(|val_ref, &m_val, &s_val| {
                    // s_val is from self.scale, which is guaranteed to be positive and finite.
                    *val_ref = (*val_ref - m_val) / s_val;
                });
        }

        // Project the centered and scaled data onto the principal components
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
            return Err("PCA::save_model: Cannot save a PCA model that is missing essential components (rotation, mean, or scale).".into());
        }
        // explained_variance being None is acceptable, for example, if the model was created
        // using `with_model` and eigenvalues were not provided or computed.
        // `load_model` contains further validation for consistency if explained_variance is Some.
        let file = File::create(path.as_ref()).map_err(|e| {
            format!(
                "PCA::save_model: Failed to create file at {:?}: {}",
                path.as_ref(),
                e
            )
        })?;
        let mut writer = BufWriter::new(file);

        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| format!("PCA::save_model: Failed to serialize PCA model: {}", e))?;
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
        let file = File::open(path.as_ref()).map_err(|e| {
            format!(
                "PCA::load_model: Failed to open file at {:?}: {}",
                path.as_ref(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);

        let pca_model: PCA =
            bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
                .map_err(|e| format!("PCA::load_model: Failed to deserialize PCA model: {}", e))?;

        let rotation = pca_model
            .rotation
            .as_ref()
            .ok_or("PCA::load_model: Loaded PCA model is missing rotation matrix.")?;
        let mean = pca_model
            .mean
            .as_ref()
            .ok_or("PCA::load_model: Loaded PCA model is missing mean vector.")?;
        let scale = pca_model
            .scale
            .as_ref()
            .ok_or("PCA::load_model: Loaded PCA model is missing scale vector.")?;

        let d_rot_features = rotation.nrows();
        let d_mean_features = mean.len();
        let d_scale_features = scale.len();

        if !(d_rot_features == d_mean_features && d_mean_features == d_scale_features) {
            if !(d_rot_features == 0
                && rotation.ncols() == 0
                && d_mean_features == 0
                && d_scale_features == 0)
            {
                return Err(format!(
                    "PCA::load_model: Loaded PCA model has inconsistent feature dimensions: rotation_features={}, mean_features={}, scale_features={}",
                    d_rot_features, d_mean_features, d_scale_features
                ).into());
            }
        }
        // Validate that loaded scale factors are positive, aligning with the contract for self.scale.
        // self.scale is expected to store sanitized, positive values (1.0 for original std devs <= SCALE_SANITIZATION_THRESHOLD, else the std dev itself).
        // Scale values must be strictly positive. EIGENVALUE_CLAMP_MIN is 0.0.
        if scale
            .iter()
            .any(|&val| !val.is_finite() || val <= EIGENVALUE_CLAMP_MIN)
        {
            return Err("PCA::load_model: Loaded PCA model's scale vector contains invalid (non-finite, zero, or negative) values. Scale values must be strictly positive.".into());
        }

        // Validate explained_variance if present
        if let Some(ev) = pca_model.explained_variance.as_ref() {
            if let Some(rot) = pca_model.rotation.as_ref() {
                if ev.len() != rot.ncols() {
                    return Err(format!(
                        "PCA::load_model: Loaded PCA model has inconsistent dimensions: explained_variance length ({}) does not match rotation matrix number of components ({}).",
                        ev.len(), rot.ncols()
                    ).into());
                }
            } else {
                // Should not happen if rotation is required for a valid model
                return Err("PCA::load_model: Loaded PCA model has explained_variance but no rotation matrix.".into());
            }
            if ev
                .iter()
                .any(|&val| !val.is_finite() || val < EIGENVALUE_CLAMP_MIN)
            {
                // Variances cannot be negative
                return Err("PCA::load_model: Loaded PCA model's explained_variance vector contains invalid (non-finite or negative) values.".into());
            }
        }
        // If rotation is Some and has components, but explained_variance is None (e.g. model from `with_model`),
        // this is an acceptable state. The `explained_variance()` accessor will simply return None.
        // If rotation itself is None or has no components (ncols == 0), then explained_variance being None is also consistent.

        Ok(pca_model)
    }
}
