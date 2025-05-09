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
        })
    }


    /// Fits the PCA model to the data using an exact covariance/Gram matrix approach.
    ///
    /// This method computes the mean, (sanitized) scaling factors, and principal axes (rotation)
    /// via an eigen-decomposition of the covariance matrix (if n_features <= n_samples)
    /// or the Gram matrix (if n_features > n_samples, the "Gram trick").
    /// The resulting principal components (columns of the rotation matrix) are normalized to unit length.
    ///
    /// **Note:** For very large datasets, `rfit` is generally recommended for better performance.
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
            } else {
                let mut top_eigvecs_owned: Vec<Array1<f64>> = Vec::with_capacity(final_rank);
                for i in 0..final_rank {
                    let mut eig_vec = eig_pairs[i].1.clone();
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
            } else {
                let mut rotation_matrix = Array2::<f64>::zeros((n_features, final_rank));
                for i in 0..final_rank {
                    let (eigval, ref u_col) = eig_pairs[i];
                    // Eigenvalue from G = X'X'^T / (N-1)
                    // V_k = (X'^T u_k) / sqrt(eigval_k_gram * (N-1)) should yield unit norm V_k.
                    
                    // Clamp eigenvalues to a small positive number to prevent division by zero or very small numbers,
                    // and to avoid skipping components with `continue`.
                    // Requested components are computed even if their variance contribution is tiny.
                    let eigval_clamped = eigval.max(1e-12); // Use max so it's at least a small positive.
                    let lam_sqrt = eigval_clamped.sqrt();

                    let mut axis_i = scaled_data_matrix.t().dot(u_col);
                    
                    // Denominator for scaling axis_i.
                    // Since n_samples >= 2 and lam_sqrt >= sqrt(1e-12), denom will be non-zero.
                    let denom = lam_sqrt * ((n_samples - 1) as f64).sqrt();
                    axis_i.mapv_inplace(|x| x / denom);
                    
                    // Explicitly re-normalize the principal axis to unit length.
                    // Clamping the eigenvalue might affect
                    // the implicit normalization of the original formula for V_k.
                    // Standard PCA components are unit vectors.
                    let norm_val = axis_i.dot(&axis_i).sqrt();
                    if norm_val > 1e-9 { // Avoid division by zero/small norm if axis_i is effectively zero
                        axis_i.mapv_inplace(|x| x / norm_val);
                    } else {
                        // If the axis vector is effectively zero (e.g., u_col was orthogonal to all data,
                        // or numerical precision resulted in a zero vector), represent it as such.
                        axis_i.fill(0.0);
                    }
                    rotation_matrix.slice_mut(s![.., i]).assign(&axis_i);
                }
                self.rotation = Some(rotation_matrix);
            }
        }
        Ok(())
    }

    /// Fits the PCA model using a randomized SVD approach.
    ///
    /// This computes the mean, (sanitized) scaling factors, and an approximate rotation
    /// (principal components). This method is designed to be more efficient for
    /// large datasets by avoiding the explicit formation of the full covariance matrix
    /// and instead working with smaller "sketched" matrices. The resulting principal
    /// components (columns of the rotation matrix) are normalized to unit length.
    ///
    /// * `x` - Input data as a 2D array (n_samples, n_features).
    /// * `n_components` - Number of principal components to keep.
    /// * `n_oversamples` - Number of additional random dimensions to sample for better accuracy
    ///                     (a common value is 5-10, e.g., `n_oversamples = 10`).
    /// * `seed` - Optional seed for reproducibility of the random number generator.
    /// * `tol` - Optional tolerance. If Some, components are kept if their singular value
    ///           is > `tol * largest_singular_value`. The number of components will be
    ///           at most `n_components`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The input matrix has zero dimensions, fewer than 2 samples.
    /// - `n_components` is 0.
    /// - SVD or QR decomposition fails during the randomized algorithm.
    pub fn rfit(
        &mut self,
        mut x: Array2<f64>, 
        n_components: usize,
        n_oversamples: usize,
        seed: Option<u64>,
        tol: Option<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err("Input matrix has zero samples or zero features.".into());
        }
        if n_samples < 2 {
            return Err("Input matrix must have at least 2 samples.".into());
        }
        if n_components == 0 {
            return Err("Number of components (n_components) must be greater than 0.".into());
        }

        let mean_vector = x.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;
        self.mean = Some(mean_vector.clone());
        x -= &mean_vector; 

        let std_dev_vector_original = x.map_axis(Axis(0), |column| column.std(0.0));
        let sanitized_scale_vector = std_dev_vector_original
            .mapv(|val| if val.abs() < 1e-9 { 1.0 } else { val });
        self.scale = Some(sanitized_scale_vector.clone()); 
        x /= &sanitized_scale_vector; 
        let a = x; 

        let k_target = n_components;
        let max_possible_rank = std::cmp::min(n_samples, n_features);
         if max_possible_rank == 0 { 
            return Err("Matrix has zero effective rank.".into());
        }

        let l_sketch_size_ideal = k_target + n_oversamples;
        let mut l_sketch_size = std::cmp::min(l_sketch_size_ideal, max_possible_rank);
        l_sketch_size = std::cmp::max(l_sketch_size, 1); 
        l_sketch_size = std::cmp::max(l_sketch_size, std::cmp::min(k_target, max_possible_rank));


        const N_POWER_ITERATIONS: usize = 2; 

        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(rand::thread_rng()).map_err(|e| e.to_string())?,
        };

        let a_op = a.t().as_standard_layout().to_owned(); 
        let (_dim_op_m_feat, dim_op_n_samp) = a_op.dim(); 

        let omega = Array2::from_shape_fn((dim_op_n_samp, l_sketch_size), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        let mut q_basis = a_op.dot(&omega); 

        if q_basis.ncols() == 0 { 
            return Err("Initial sketch Q_basis has zero columns. This might happen if l_sketch_size is zero or A_op leads to an all-zero sketch.".into());
        }
        q_basis = q_basis.qr()?.0; 

        for _ in 0..N_POWER_ITERATIONS {
            if q_basis.ncols() == 0 { break; } 
            let w_intermediate = a_op.t().dot(&q_basis); 
            if w_intermediate.ncols() == 0 { break; }
            let w_ortho = w_intermediate.qr()?.0;

            if w_ortho.ncols() == 0 { break; }
            let z_intermediate = a_op.dot(&w_ortho);
            if z_intermediate.ncols() == 0 { break; }
            q_basis = z_intermediate.qr()?.0;
        }
        
        let b_projected = q_basis.t().dot(&a_op); 

        let (u_b_opt, s_b_singular_values, _vt_b_opt) = b_projected.svd_into(true, false)
            .map_err(|e| format!("SVD of small matrix B failed: {}", e))?;
        let u_b = u_b_opt.ok_or("SVD U_B not computed from small matrix B")?;

        let mut rotation_full_l = q_basis.dot(&u_b); 

        for mut col in rotation_full_l.columns_mut() {
            let norm = col.dot(&col).sqrt();
            if norm > 1e-9 {
                col.mapv_inplace(|v| v / norm);
            } else {
                col.fill(0.0); 
            }
        }
        
        let effective_k_target = std::cmp::min(k_target, rotation_full_l.ncols());
        let mut num_components_to_keep = effective_k_target;

        if let Some(t_val) = tol {
            if !s_b_singular_values.is_empty() && t_val > 0.0 && t_val <= 1.0 { // Ensure tol is a valid fraction
                let largest_s_val = s_b_singular_values[0];
                if largest_s_val > 1e-9 {
                    let threshold_s_val = t_val * largest_s_val;
                    let rank_by_tol = s_b_singular_values.iter()
                                        .take(effective_k_target) 
                                        .take_while(|&&s| s > threshold_s_val)
                                        .count();
                    num_components_to_keep = rank_by_tol;
                } else { 
                    num_components_to_keep = 0;
                }
            }
        }
        num_components_to_keep = std::cmp::min(num_components_to_keep, rotation_full_l.ncols());

        if num_components_to_keep == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
        } else {
            let final_rotation_matrix = rotation_full_l.slice(s![.., ..num_components_to_keep]).to_owned();
            self.rotation = Some(final_rotation_matrix);
        }

        Ok(())
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
    /// The model must be fitted or loaded (i.e., contain rotation, mean, and scale) for saving.
    ///
    /// * `path` - The file path to save the model to.
    ///
    /// # Errors
    /// Returns an error if the model is incomplete or if file I/O or serialization fails.
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        if self.rotation.is_none() || self.mean.is_none() || self.scale.is_none() {
            return Err("Cannot save an incomplete or unfitted PCA model.".into());
        }
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
            data_matrix.clone(),
            n_components,
            5, // oversampling parameter
            Some(42), // seed
            None, // no variance tolerance
        ) {
            Ok(()) => {
                println!("PCA computation successful");
                
                // Transform data to get PC coordinates
                let transformed = match pca.transform(data_matrix.clone()) {
                    Ok(t) => t,
                    Err(e) => {
                        panic!("PCA transformation failed: {}", e);
                    }
                };
                
                // Check for NaN values
                let nan_count_unfiltered = transformed.iter().filter(|&&x| x.is_nan()).count();
                let total_values = transformed.nrows() * transformed.ncols();
                
                println!("NaN check: {}/{} values are NaN ({:.2}%)", 
                         nan_count_unfiltered, total_values, 
                         100.0 * nan_count_unfiltered as f64 / total_values as f64);
                
                // Print first few PC values for inspection
                println!("First 3 rows of PC values:");
                for i in 0..3 {
                    print!("Row {}: ", i);
                    for j in 0..n_components {
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
                        filtered_matrix.clone(),
                        n_components,
                        5,
                        Some(42),
                        None,
                    ) {
                        Ok(()) => {
                            let transformed_filtered = pca_filtered.transform(filtered_matrix).unwrap();
                            let nan_count_filtered = transformed_filtered.iter().filter(|&&x| x.is_nan()).count();
                            println!("Filtered PCA NaN check: {}/{} values are NaN", 
                                    nan_count_filtered, transformed_filtered.len());
                            
                            // Print the first 3 rows of filtered PC values to demonstrate they are valid
                            println!("First 3 rows of FILTERED PC values:");
                            for i in 0..3 {
                                print!("Row {}: ", i);
                                for j in 0..n_components {
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

        // Fit PCA with rfit
        let start_pca = Instant::now();
        pca.rfit(data.clone(), n_components, n_oversamples, seed, None)?;
        let pca_duration = start_pca.elapsed();
        println!("Randomized PCA fit completed in {:.2?}", pca_duration);

        // Apply PCA transformation
        let start_transform = Instant::now();
        let transformed = pca.transform(data)?;
        let transform_duration = start_transform.elapsed();
        println!("PCA transform completed in {:.2?}", transform_duration);

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
        rust_pca_rfit.rfit(data.clone(), n_components, 5, Some(42_u64), None)?;
        let rust_transformed_rfit = rust_pca_rfit.transform(data.clone())?;

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
        pca_original.rfit(data.clone(), n_components_to_fit, 10, Some(123), None)?; // Using more oversamples

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
        if is_rpca {
            pca.rfit(input.clone(), n_components, oversamples, seed, None)
                .unwrap();
        } else {
            pca.fit(input.clone(), None).unwrap();
        }
        let rust_transformed = pca.transform(input.clone()).unwrap();

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
        if use_rfit {
            pca.rfit(data.clone(), n_components, n_oversamples, Some(seed), None)
                .expect("rfit failed");
        } else {
            pca.fit(data.clone(), None).expect("fit failed");
        }

        // Also transform the data so the entire pipeline is included
        let transformed = pca.transform(data.clone()).expect("transform failed");
        assert_eq!(transformed.nrows(), data.nrows());

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


#[cfg(test)]
mod hgdp_1kgp_data_pca_tests {
    use super::PCA; // PCA struct is in the parent module or crate root accessible via `super`

    // Ndarray and Rand crates for matrix operations and data generation
    use ndarray::Array2;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // Standard library imports for file and I/O operations
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader, Write, Read};
    use std::path::{Path, PathBuf};
    use std::error::Error;

    // Crates for HTTP, compression, archiving, temp files, and directories
    use reqwest::blocking::Client as ReqwestBlockingClient;
    use zstd::Decoder as ZstdDecoder;
    use tar::Archive as TarArchive;
    use tempfile::TempDir;
    use dirs; // For user's home/cache directory

    const FILE_URL: &str = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst";
    const ARCHIVE_FILENAME: &str = "pgsc_HGDP+1kGP_v1.tar.zst";
    const FAM_FILE_IN_ARCHIVE: &str = "pgsc_HGDP+1kGP.fam"; // Name of the .fam file inside the tarball
    const BIM_FILE_IN_ARCHIVE: &str = "pgsc_HGDP+1kGP.bim"; // Name of the .bim file inside the tarball

    fn download_if_not_present(
        url: &str,
        output_directory: &Path,
        output_filename: &str,
    ) -> Result<PathBuf, Box<dyn Error>> {
        let output_path = output_directory.join(output_filename);
        if output_path.exists() {
            println!("Archive {} already exists in {:?}. Skipping download.", output_filename, output_directory);
            return Ok(output_path);
        }

        println!("Downloading (Rust-native) {} from {} to {:?}...", output_filename, url, output_directory);
        fs::create_dir_all(output_directory)?;

        let client = ReqwestBlockingClient::builder()
            .timeout(None) // No timeout for potentially large downloads
            .build()?;
        
        let mut response = client.get(url).send()?;

        if !response.status().is_success() {
            return Err(format!("Failed to download file (reqwest): HTTP {}", response.status()).into());
        }

        let mut dest_file = File::create(&output_path)
            .map_err(|e| format!("Failed to create destination file {:?}: {}", output_path, e))?;
        
        let content_length = response.content_length().unwrap_or(0);
        let mut downloaded_bytes: u64 = 0;
        let mut buffer = vec![0; 16 * 1024]; // 16KB buffer, increased for potentially better performance

        println!("Download size (reqwest): {} bytes", content_length);
        let mut last_reported_mb = 0;
        let report_interval_mb = 20; // Report every 20MB

        loop {
            let bytes_read = response.read(&mut buffer)?;
            if bytes_read == 0 {
                break; // End of stream
            }
            dest_file.write_all(&buffer[..bytes_read])?;
            downloaded_bytes += bytes_read as u64;

            let current_mb = downloaded_bytes / (1024 * 1024);
            if content_length > 0 { // Only show progress if total size is known
                if current_mb > last_reported_mb && (current_mb % report_interval_mb == 0 || current_mb * 1024 * 1024 >= content_length) {
                    let total_mb_display = content_length / (1024 * 1024);
                    println!("Downloaded (reqwest): {} / {} MB ({:.2}%)", 
                             current_mb, 
                             total_mb_display,
                             (downloaded_bytes as f64 * 100.0) / content_length as f64);
                    last_reported_mb = current_mb;
                }
            } else if current_mb > last_reported_mb && current_mb % report_interval_mb == 0 {
                 println!("Downloaded (reqwest): {} MB (total size unknown)", current_mb);
                 last_reported_mb = current_mb;
            }
        }
        println!("Rust-native download complete: {:?}", output_path);
        Ok(output_path)
    }

    fn extract_target_files(
        archive_path: &Path,
        extraction_dir: &Path,
        target_files_in_archive: &[&str],
    ) -> Result<Vec<PathBuf>, Box<dyn Error>> {
        println!("Ensuring files {:?} are extracted from {:?} to {:?}...", target_files_in_archive, archive_path, extraction_dir);
        fs::create_dir_all(extraction_dir)?;

        let mut paths_to_return = Vec::new();
        let mut files_needing_extraction_from_tar = Vec::new();

        // Check which files already exist and which need to be extracted
        for target_filename_in_archive in target_files_in_archive {
            let expected_output_path = extraction_dir.join(target_filename_in_archive);
            if expected_output_path.exists() {
                println!("File {} already exists in extraction dir: {:?}", target_filename_in_archive, expected_output_path);
                paths_to_return.push(expected_output_path);
            } else {
                files_needing_extraction_from_tar.push(*target_filename_in_archive);
            }
        }

        // If all files that were sought are already in paths_to_return, no extraction needed for them
        if files_needing_extraction_from_tar.is_empty() && paths_to_return.len() == target_files_in_archive.len() {
            println!("All target files already present in extraction directory.");
            return Ok(paths_to_return);
        }
        
        println!("Files to attempt extraction for: {:?}", files_needing_extraction_from_tar);

        let archive_file = File::open(archive_path)
            .map_err(|e| format!("Failed to open archive file {:?}: {}", archive_path, e))?;
        let zstd_decoder = ZstdDecoder::new(BufReader::new(archive_file))
            .map_err(|e| format!("Failed to create zstd decoder for {:?}: {}", archive_path, e))?;
        let mut tar_archive = TarArchive::new(BufReader::new(zstd_decoder));

        let mut extracted_filenames_this_run = std::collections::HashSet::new();

        for entry_result in tar_archive.entries()? {
            let mut entry = entry_result?;
            let entry_path_obj = entry.path()?; 
            
            let entry_filename = match entry_path_obj.file_name() {
                Some(name) => name.to_string_lossy().into_owned(),
                None => continue, // Skip if no filename (e.g. a directory entry itself)
            };

            // Check if this entry's filename is one we're looking to extract
            if files_needing_extraction_from_tar.contains(&entry_filename.as_str()) {
                let final_extraction_path = extraction_dir.join(&entry_filename);
                // Unpack even if it exists, to ensure it's from the current archive version if logic implies overwrite
                // Or, add a check here: if !final_extraction_path.exists()
                if !final_extraction_path.exists() {
                    if let Some(parent_dir) = final_extraction_path.parent() {
                        fs::create_dir_all(parent_dir)?;
                    }
                    entry.unpack(&final_extraction_path)?;
                    println!("Extracted: {:?}", final_extraction_path);
                    extracted_filenames_this_run.insert(entry_filename.clone());
                } else {
                    println!("File {:?} already existed, not re-extracting.", final_extraction_path);
                    extracted_filenames_this_run.insert(entry_filename.clone()); // Consider it "handled"
                }
            }
        }
        
        // Rebuild paths_to_return to ensure it only contains paths to files that now exist
        paths_to_return.clear();
        for target_filename_in_archive in target_files_in_archive {
            let expected_output_path = extraction_dir.join(target_filename_in_archive);
            if expected_output_path.exists() {
                paths_to_return.push(expected_output_path);
            } else {
                 // This means a file listed in target_files_in_archive was not found/extracted
                return Err(format!("Critical error: File {} was expected but not found after extraction attempt: {:?}", target_filename_in_archive, extraction_dir.join(target_filename_in_archive)).into());
            }
        }
        
        if paths_to_return.len() != target_files_in_archive.len() {
             let missing_files: Vec<&str> = target_files_in_archive.iter()
                .filter(|f_name| !paths_to_return.iter().any(|p| p.file_name().unwrap_or_default() == **f_name))
                .map(|s| *s)
                .collect();
            return Err(format!("Not all target files were successfully made available. Missing: {:?}", missing_files).into());
        }

        Ok(paths_to_return)
    }

    fn read_plink_dimensions(fam_path: &Path, bim_path: &Path) -> Result<(usize, usize), Box<dyn Error>> {
        let fam_file = File::open(fam_path)
            .map_err(|e| format!("Failed to open FAM file {:?}: {}", fam_path, e))?;
        let n_samples = BufReader::new(fam_file).lines().count();

        let bim_file = File::open(bim_path)
            .map_err(|e| format!("Failed to open BIM file {:?}: {}", bim_path, e))?;
        let n_variants = BufReader::new(bim_file).lines().count();
        
        if n_samples == 0 { return Err(format!("FAM file {:?} is empty or unreadable, 0 samples found.", fam_path).into()); }
        if n_variants == 0 { return Err(format!("BIM file {:?} is empty or unreadable, 0 variants found.", bim_path).into()); }

        Ok((n_samples, n_variants))
    }

    fn generate_dummy_genotype_matrix(
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n_samples, n_features), |_| {
            rng.gen_range(0..=2) as f64 
        })
    }

    #[test]
    fn test_pca_on_downloaded_ebi_data() -> Result<(), Box<dyn Error>> {
        println!("\n\nWARNING: This test will download a ~1.1 GB file if not already present.");
        println!("It may take a significant amount of time and network bandwidth.");
        println!("Using Rust-native reqwest for download.");
        println!("Starting PCA test on downloaded EBI data...\n");

        let home_dir = dirs::home_dir()
            .ok_or_else(|| "Could not find home directory to create download cache".to_string())?;
        let persistent_download_dir = home_dir.join(".cache").join("rust_pca_test_downloads");
        
        // Extractions will be in a temporary directory that cleans up afterwards
        let extraction_base_temp_dir = TempDir::new()
            .map_err(|e| format!("Failed to create base temp dir for extraction: {}", e))?;
        let extraction_dir = extraction_base_temp_dir.path().join("ebi_extracted_data");
        
        fs::create_dir_all(&persistent_download_dir)
            .map_err(|e| format!("Failed to create persistent download directory {:?}: {}", persistent_download_dir, e))?;
        fs::create_dir_all(&extraction_dir)
            .map_err(|e| format!("Failed to create temporary extraction directory {:?}: {}", extraction_dir, e))?;

        // 1. Download the archive (if not present) using pure Rust
        let archive_path = download_if_not_present(FILE_URL, &persistent_download_dir, ARCHIVE_FILENAME)?;
        assert!(archive_path.exists(), "Archive file {:?} should exist after download step.", archive_path);

        // 2. Extract .fam and .bim files
        let target_metadata_files = [FAM_FILE_IN_ARCHIVE, BIM_FILE_IN_ARCHIVE];
        let extracted_files = extract_target_files(&archive_path, &extraction_dir, &target_metadata_files)?;
        
        let fam_file_path = extracted_files.iter()
            .find(|p| p.file_name().map_or(false, |name| name == FAM_FILE_IN_ARCHIVE))
            .ok_or_else(|| format!("{} file path not found in extracted files list", FAM_FILE_IN_ARCHIVE))?;
        let bim_file_path = extracted_files.iter()
            .find(|p| p.file_name().map_or(false, |name| name == BIM_FILE_IN_ARCHIVE))
            .ok_or_else(|| format!("{} file path not found in extracted files list", BIM_FILE_IN_ARCHIVE))?;

        assert!(fam_file_path.exists(), ".fam file ({:?}) not extracted or path incorrect.", fam_file_path);
        assert!(bim_file_path.exists(), ".bim file ({:?}) not extracted or path incorrect.", bim_file_path);

        // 3. Read dimensions from .fam and .bim
        let (actual_n_samples, actual_n_variants) = read_plink_dimensions(fam_file_path, bim_file_path)?;
        println!("Actual dimensions from dataset: {} samples, {} variants.", actual_n_samples, actual_n_variants);

        // 4. Generate a dummy genotype matrix with capped dimensions for the test
        let test_n_samples = std::cmp::min(actual_n_samples, 150); 
        let test_n_variants = std::cmp::min(actual_n_variants, 500);
        println!("Using capped dimensions for PCA test: {} samples, {} variants.", test_n_samples, test_n_variants);

        if test_n_samples < 2 {
             return Err(format!("Not enough samples ({}) to run PCA (min 2 required by PCA fit logic).", test_n_samples).into());
        }
        if test_n_variants == 0 {
            return Err(format!("Not enough variants ({}) to run PCA (min 1 required).", test_n_variants).into());
        }

        let genotype_matrix = generate_dummy_genotype_matrix(test_n_samples, test_n_variants, 42);

        // 5. Run PCA
        let mut pca_model = PCA::new();
        let n_components_to_compute = std::cmp::min(std::cmp::min(test_n_samples, test_n_variants), 10).max(1);
        let n_oversamples = 5; // Typical value for randomized PCA
        let pca_seed = Some(42u64);
        
        println!("Fitting PCA model (rfit) with {} components...", n_components_to_compute);
        pca_model.rfit(
            genotype_matrix.clone(), 
            n_components_to_compute,
            n_oversamples,
            pca_seed,
            None, // No specific variance tolerance for component selection
        )?;
        println!("PCA model fitted.");

        // 6. Transform data
        println!("Transforming data...");
        let transformed_data = pca_model.transform(genotype_matrix)?; // Pass original matrix again
        println!("Data transformed.");

        // 7. Assertions
        assert_eq!(transformed_data.nrows(), test_n_samples, "Transformed data should have same number of rows as input samples.");
        assert_eq!(transformed_data.ncols(), n_components_to_compute, "Transformed data should have k_components columns.");
        assert!(
            transformed_data.iter().all(|&val| val.is_finite()),
            "Transformed data contains non-finite values (NaN or infinity)."
        );

        println!("\nPCA test on downloaded EBI data (using dummy matrix with capped real dimensions) completed successfully.");
        Ok(())
    }
}
