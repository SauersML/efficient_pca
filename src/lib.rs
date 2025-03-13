// Principal component analysis (PCA)

#![doc = include_str!("../README.md")]

use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::eigh::Eigh;
use ndarray_linalg::svd::SVD;
use ndarray_linalg::QR;
use ndarray_linalg::UPLO;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;
use rand::TryRngCore;
use std::error::Error;

/// Principal component analysis (PCA) structure
pub struct PCA {
    /// the rotation matrix
    rotation: Option<Array2<f64>>,
    /// mean of input data
    mean: Option<Array1<f64>>,
    /// scale of input data
    scale: Option<Array1<f64>>,
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

impl PCA {
    /// Create a new PCA struct with default values
    ///
    /// # Examples
    ///
    /// ```
    /// use efficient_pca::PCA;
    /// let pca = PCA::new();
    /// ```
    pub fn new() -> Self {
        Self {
            rotation: None,
            mean: None,
            scale: None,
        }
    }

    /// Fit the PCA rotation to the data **using the covariance approach**.
    ///
    /// This method computes the mean, scaling, and principal axes (rotation)
    /// via an eigen-decomposition of the covariance matrix \( X \cdot X^T \).
    /// This approach is especially suitable when the number of features (columns)
    /// is much larger than the number of samples (rows).
    ///
    /// * `data_matrix` - Input data as a 2D array, shape (n_samples, n_features).
    /// * `tolerance` - Tolerance for excluding low-variance components
    ///   (fraction of the largest eigenvalue). If `None`, all components are kept.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input matrix has fewer than 2 rows.
    /// - Eigen-decomposition fails (very unlikely for a well-formed covariance).
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use efficient_pca::PCA;
    ///
    /// let data = array![
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ];
    ///
    /// let mut pca = PCA::new();
    /// pca.fit(data, None).unwrap();
    /// ```
    pub fn fit(
        &mut self,
        mut data_matrix: Array2<f64>,
        tolerance: Option<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_samples = data_matrix.nrows();
        let n_features = data_matrix.ncols();

        if n_samples < 2 {
            return Err("Input matrix must have at least 2 rows.".into());
        }

        // 1) Center and scale the data
        let mean_vector = data_matrix
            .mean_axis(Axis(0))
            .ok_or("Failed to compute mean of the data.")?;
        self.mean = Some(mean_vector.clone());
        data_matrix -= &mean_vector;

        let std_dev_vector = data_matrix.map_axis(Axis(0), |column| column.std(0.0));
        self.scale = Some(std_dev_vector.clone());
        data_matrix /= &std_dev_vector.mapv(|val| if val != 0.0 { val } else { 1.0 });

        // Decide which covariance trick to use
        //  - If p <= n, use the f×f covariance = (X^T X)/(n-1), then eigendecompose it
        //  - If p > n, use the n×n Gram matrix = (X X^T)/(n-1), then map to feature space
        if n_features <= n_samples {
            // ================================
            // (A) Standard f×f covariance
            // ================================
            let mut cov_matrix = data_matrix.t().dot(&data_matrix);
            cov_matrix /= (n_samples - 1) as f64;

            // Eigen-decomposition (p×p)
            let (vals, vecs) = cov_matrix
                .eigh(UPLO::Upper)
                .map_err(|_| "Eigen decomposition of covariance failed.")?;

            // Sort descending by eigenvalue
            let mut eig_pairs: Vec<(f64, Array1<f64>)> = vals
                .iter()
                .cloned()
                .zip(vecs.columns().into_iter().map(|col| col.to_owned()))
                .collect();
            eig_pairs.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

            // Apply tolerance if needed
            let largest = eig_pairs.get(0).map_or(0.0, |(v, _)| *v);
            let rank_by_tol = if let Some(tol) = tolerance {
                let thresh = largest * tol;
                eig_pairs
                    .iter()
                    .take_while(|(val, _)| *val > thresh)
                    .count()
            } else {
                eig_pairs.len()
            };
            let final_rank = std::cmp::min(rank_by_tol, n_features);

            // Gather top components
            let mut top_eigvecs = Vec::with_capacity(final_rank);
            for i in 0..final_rank {
                let (_, ref evec) = eig_pairs[i];
                top_eigvecs.push(evec.clone());
            }
            let top_eigvecs = ndarray::stack(
                Axis(1),
                &top_eigvecs.iter().map(|v| v.view()).collect::<Vec<_>>(),
            )?;

            // These vectors are the principal axes in feature space
            self.rotation = Some(top_eigvecs);

            // So that each column is unit length
            if let Some(ref mut rot) = self.rotation {
                for i in 0..rot.ncols() {
                    let mut col_i = rot.slice_mut(s![.., i]);
                    let norm_i = col_i.dot(&col_i).sqrt();
                    if norm_i > 1e-12 {
                        col_i.mapv_inplace(|x| x / norm_i);
                    }
                }
            }
        } else {
            // ==========================================
            // (B) Gram trick: n×n covariance
            // ==========================================
            let mut gram_matrix = data_matrix.dot(&data_matrix.t());
            gram_matrix /= (n_samples - 1) as f64;

            // Eigen-decompose (n×n)
            let (vals, vecs) = gram_matrix
                .eigh(UPLO::Upper)
                .map_err(|_| "Eigen decomposition of Gram matrix failed.")?;

            // Sort descending
            let mut eig_pairs: Vec<(f64, Array1<f64>)> = vals
                .iter()
                .cloned()
                .zip(vecs.columns().into_iter().map(|col| col.to_owned()))
                .collect();
            eig_pairs.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

            // Tolerance
            let largest = eig_pairs.get(0).map_or(0.0, |(v, _)| *v);
            let rank_by_tol = if let Some(tol) = tolerance {
                let thresh = largest * tol;
                eig_pairs
                    .iter()
                    .take_while(|(val, _)| *val > thresh)
                    .count()
            } else {
                eig_pairs.len()
            };
            let final_rank = std::cmp::min(rank_by_tol, n_samples);

            // Build the principal axes in feature space
            // rotation_matrix has shape (p, final_rank)
            let mut rotation_matrix = Array2::<f64>::zeros((n_features, final_rank));

            for i in 0..final_rank {
                let (eigval, ref u_col) = eig_pairs[i];
                // If eigenvalue is extremely small or negative (rounding), clamp:
                let lam = if eigval > 1e-12 { eigval } else { 1e-12 };
                // Feature-space vector = X^T * u / sqrt(lam)
                // shape => (p,)
                let mut axis_i = data_matrix.t().dot(u_col);
                axis_i.mapv_inplace(|x| x / lam.sqrt());

                // extra division by sqrt(n-1):
                axis_i.mapv_inplace(|x| x / ((n_samples - 1) as f64).sqrt());

                // then normalize this axis to length 1:
                // let norm_i = axis_i.dot(&axis_i).sqrt();
                // axis_i.mapv_inplace(|x| x / norm_i);
                // actually no

                // Put it as the i-th column in rotation_matrix
                rotation_matrix.slice_mut(s![.., i]).assign(&axis_i);
            }
            self.rotation = Some(rotation_matrix);
        }

        Ok(())
    }

    /// Use randomized SVD to fit a PCA rotation to the data
    ///
    /// This computes the mean, scaling and rotation to apply PCA
    /// to the input data matrix.
    ///
    /// * `x` - Input data as a 2D array
    /// * `n_components` - Number of components to keep
    /// * `n_oversamples` - Number of oversampled dimensions (for rSVD)
    /// * `tol` - Tolerance for excluding low variance components.
    ///           If None, all components are kept.
    ///
    /// # Errors
    ///
    /// Returns an error if the input matrix has fewer than 2 rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use efficient_pca::PCA;
    ///
    /// let x = array![[1.0, 2.0], [3.0, 4.0]];
    /// let mut pca = PCA::new();
    /// pca.rfit(x, 1, 0, None, None).unwrap();
    /// ```
    /// `rsvd` internally calls `.svd(true, true)`.
    pub fn rfit(
        &mut self,
        mut x: Array2<f64>,
        n_components: usize,
        n_oversamples: usize,
        seed: Option<u64>,
        tol: Option<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let n = x.nrows();
        if n < 2 {
            return Err("Input matrix must have at least 2 rows.".into());
        }

        // Compute mean for centering
        let mean = x.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;
        self.mean = Some(mean.clone());
        x -= &mean;

        // Compute scale
        let std_dev = x.map_axis(Axis(0), |v| v.std(0.0));
        self.scale = Some(std_dev.clone());
        x /= &std_dev.mapv(|v| if v != 0. { v } else { 1. });

        // Determine effective number of components (can't exceed number of samples)
        let k = std::cmp::min(n_components, n);

        // COVARIANCE TRICK: Compute covariance matrix (n×n) instead of working with original (n×p) matrix
        // This is much more efficient when p >> n
        let cov_matrix = x.dot(&x.t()) / (n as f64 - 1.0).max(1.0);

        // Use randomized SVD on the covariance matrix
        // For a symmetric matrix like covariance matrix, u contains the eigenvectors
        let (u, s, _) = rsvd(&cov_matrix, k, n_oversamples, seed);

        // Extract singular values
        let singular_values = s.diag().to_owned();

        // Apply tolerance if specified
        let final_k = if let Some(t) = tol {
            let threshold = singular_values[0] * t;
            let rank = singular_values
                .iter()
                .take(k)
                .take_while(|&v| *v > threshold)
                .count();
            rank
        } else {
            k
        };

        // Extract the top eigenvectors
        let top_eigenvectors = u.slice(s![.., ..final_k]).to_owned();

        // Map the eigenvectors (u) back to feature space
        let mut rotation_matrix = x.t().dot(&top_eigenvectors);

        // For each column, divide by sqrt(eigval), then / sqrt(n-1), then unit‐normalize.
        // singular_values[i] is the eigenvalue (already s[i], from the n×n covariance)
        for i in 0..final_k {
            let lam = singular_values[i].max(1e-12);

            // slice the i-th column in "rotation_matrix"
            let mut col_i = rotation_matrix.slice_mut(s![.., i]);

            // 1) / sqrt(lambda)
            col_i.mapv_inplace(|v| v / lam.sqrt());

            // 2) / sqrt(n-1)
            col_i.mapv_inplace(|v| v / ((n as f64 - 1.0).sqrt()));

            // 3) final unit length
            // let norm_i = col_i.dot(&col_i).sqrt();
            // if norm_i > 1e-12 {
            //     col_i.mapv_inplace(|v| v / norm_i);
            // }
            //  actually no
        }
        self.rotation = Some(rotation_matrix);
        Ok(())
    }

    /// Apply the PCA rotation to the data
    ///
    /// This projects the data into the PCA space using the
    /// previously computed rotation, mean and scale.
    ///
    /// * `x` - Input data to transform
    ///
    /// # Errors
    ///
    /// Returns an error if PCA has not been fitted yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use efficient_pca::PCA;
    ///
    /// let x = array![[1.0, 2.0],[3.0, 4.0]];
    /// let mut pca = PCA::new();
    /// pca.fit(x.clone(), None).unwrap();
    /// pca.transform(x).unwrap();
    /// ```
    pub fn transform(&self, mut x: Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        match (&self.rotation, &self.mean, &self.scale) {
            (Some(rotation), Some(mean), Some(scale)) => {
                x -= mean;
                x /= scale;
                Ok(x.dot(rotation))
            }
            _ => Err("PCA not fitted yet.".into()),
        }
    }
}

// =================================================================================
// rSVD implementation copied from: https://github.com/ekg/rsvd/blob/main/src/lib.rs

/// Calculate a randomized SVD approximation of a matrix.
///
/// # Arguments
///
/// * `input` - The matrix to compute the randomized SVD for.
/// * `k` - The target rank for the approximation.
/// * `p` - The oversampling parameter.
///
/// # Returns
///
/// A tuple `(u, s, vt)` containing:
///
/// * `u` - The left singular vectors.
/// * `s` - The singular values.
/// * `vt` - The right singular vectors.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use rsvd::rsvd;
///
/// let a = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
/// let (u, s, vt) = rsvd(&a, 2, 1, None);
/// ```
pub fn rsvd(
    input: &Array2<f64>,
    k: usize,
    p: usize,
    seed: Option<u64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    //let m = input.shape()[0];
    let n = input.shape()[1];

    // Handle the seed, which could be None
    // If it's None, we should use whatever default seeding is used by the RNG
    let rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => match ChaCha8Rng::from_rng(&mut rand::rng()) {
            Ok(rng) => rng,
            Err(e) => panic!("Failed to create ChaCha8Rng from RNG: {:?}", e),
        },
    };

    // Generate Gaussian random test matrix
    let l = k + p; // Oversampling
    let omega = {
        let vec = rng
            .sample_iter(Normal::new(0.0, 1.0).unwrap())
            .take(l * n)
            .collect::<Vec<_>>();
        ndarray::Array::from_shape_vec((n, l), vec).unwrap()
    };

    // Form sample matrix Y
    let y = input.dot(&omega);

    // Orthogonalize Y
    let (q, _) = y.qr().unwrap();

    // Project input to lower dimension
    let b = q.t().dot(input);

    // Compute SVD of small matrix B
    let (Some(u), s, Some(vt)) = b.svd(true, true).unwrap() else {
        panic!("SVD failed");
    };

    // Convert s to an Array2<f64>
    let s = Array2::from_diag(&s);

    // Return truncated SVD
    (q.dot(&u), s, vt)
}

#[cfg(test)]
mod genome_tests {
    use super::*;
    use ndarray::{Array2, ArrayView1};
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;
    use std::io::Write;
    use std::process::Command;
    use std::time::Instant;
    use sysinfo::System;
    use tempfile::NamedTempFile;

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
        let process_after = sys.process(pid).expect("Unable to get current process after test");
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
        println!("\n[Controlled Structure Test] Generating matrix with exactly 3 real components...");
        
        // Use same dimensions as the original genotype test
        let n_samples = 88;
        let n_variants = 10000;
        let n_real_components = 3;  // Exactly 3 components represent true structure
        let signal_strength = [50.0, 20.0, 10.0];  // Stronger to weaker signals for each component
        
        // Set random seed for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        // Create orthogonal basis vectors for the signal components (QR decomposition)
        let random_basis = Array2::<f64>::from_shape_fn((n_samples, n_real_components), 
                                                       |_| rng.gen_range(-1.0..1.0));
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
        
        println!("[Controlled Test] Created data with {} real components and pure noise", n_real_components);
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
                let line = row.iter()
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
        eigenvalues.as_slice_mut().unwrap().sort_by(|a, b| b.partial_cmp(a).unwrap());
        let total_variance: f64 = eigenvalues.iter().take(n_components).sum();
        
        println!("\n[Comparison] Explained Variance:");
        println!("Component | Rust Eigenvalue |  %   | Status");
        println!("---------+----------------+------+--------");
        for i in 0..n_components {
            let variance_pct = (eigenvalues[i] / total_variance) * 100.0;
            let status = if i < n_real_components { "REAL SIGNAL" } else { "PURE NOISE" };
            println!("    PC{:<2}  | {:>14.2} | {:>4.1}% | {}", 
                    i+1, eigenvalues[i], variance_pct, status);
        }
        
        println!("\nTotal variance: {:.2}", total_variance);
        println!("First {} PCs capture real structure, remaining PCs are pure noise", n_real_components);
        
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
                    println!("    PC{:<2}  | {:>10.4} | >={:.2}     | ✗ FAILED", 
                           pc_idx+1, abs_correlation, threshold);
                } else {
                    println!("    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ PASSED", 
                           pc_idx+1, abs_correlation, threshold);
                }
            } else {
                println!("    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ IGNORED", 
                       pc_idx+1, abs_correlation, 0.0);
            }
        }
        
        assert!(all_real_components_match_fit, "Real signal components do not match for fit method");
        
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
                    println!("    PC{:<2}  | {:>10.4} | >={:.2}     | ✗ FAILED", 
                           pc_idx+1, abs_correlation, threshold);
                } else {
                    println!("    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ PASSED", 
                           pc_idx+1, abs_correlation, threshold);
                }
            } else {
                println!("    PC{:<2}  | {:>10.4} | >={:.2}     | ✓ IGNORED", 
                       pc_idx+1, abs_correlation, 0.0);
            }
        }
        
        assert!(all_real_components_match_rfit, "Real signal components do not match for rfit method");
        
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
                        Err(_) => 0.0
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
            println!("[Rust Debug] Parsing into {} rows x {} cols.", row_count, col_count);

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
                            },
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
            println!("[Rust Debug] Final parsed array shape = ({}, {}).", arr.nrows(), arr.ncols());
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

// Note: memory benchmarks are not accurate
#[cfg(test)]
mod rsvd_tests {
    use super::*;
    use ndarray::Array2;

    // input a dimension, a seed, and a tolerance
    // we make a random matrix to match
    // then check if the rsvd is within a tolerance of the actual svd
    fn test_rsvd(m: usize, n: usize, k: usize, p: usize, seed: u64, tol: f64) {
        // Generate random matrix
        // use seeded RNG
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let a = Array2::from_shape_fn((m, n), |_| rng.gen::<f64>());

        // Compute rank approximation
        let (u, s, vt) = rsvd(&a, k, p, Some(1337));

        let (Some(u2), s2, Some(vt2)) = a.svd(true, true).unwrap() else {
            panic!("SVD failed");
        };

        // convert s2 to a vector and diagonalize
        let s2 = Array2::from_diag(&s2);

        // if we have a rank k approximation, the u and u2 matrices and singular values are not comparable
        // so we can skip the comparison and focus on vt
        if k >= m {
            assert!(equivalent(&u, &u2, tol));
            assert!(equivalent(&s, &s2, tol));
        } else {
            //vt = vt.slice_move(s![..k, ..]);
            //vt2 = vt2.slice_move(s![..k, ..]);
        }

        // display the matrices for each method
        //eprintln!("u: \n{:?}", u);
        //eprintln!("s: \n{:?}", s);
        //eprintln!("vt: \n{:?}", vt);
        //eprintln!("u2: \n{:?}", u2);
        //eprintln!("s2: \n{:?}", s2);
        //eprintln!("vt2: \n{:?}", vt2);

        assert!(equivalent(&vt, &vt2, tol));
    }

    fn equivalent(a: &Array2<f64>, b: &Array2<f64>, e: f64) -> bool {
        let a = a.clone().mapv_into(f64::abs);
        let b = b.clone().mapv_into(f64::abs);
        // sum of absolute differences
        let diff = a - b;
        // average difference per cell
        let avg = diff.sum() / (diff.len() as f64);
        avg < e
    }

    // test 2x2 matrix
    #[test]
    fn test_rsvd_2x2() {
        for i in 0..20 {
            test_rsvd(2, 2, 1, 1, i, 1e-2);
        }
    }

    // test 3x3 matrix
    #[test]
    fn test_rsvd_3x3() {
        for i in 0..20 {
            test_rsvd(3, 3, 3, 1, i, 1e-2);
        }
    }

    // test 5x5 matrix
    #[test]
    fn test_rsvd_5x5() {
        for i in 0..20 {
            test_rsvd(5, 5, 5, 1, i, 1e-2);
        }
    }

    // test 10x10 matrix with 5 singular values
    #[test]
    fn test_rsvd_10x10_k5() {
        for i in 0..20 {
            test_rsvd(10, 10, 5, 1, i, 0.1);
        }
    }

    // test 10x10 matrix with 8 singular values
    #[test]
    fn test_rsvd_10x10_k8() {
        for i in 0..20 {
            test_rsvd(10, 10, 5, 1, i, 0.1);
        }
    }

    // test 100x100 matrix k=10
    #[test]
    fn test_rsvd_100x100_k10() {
        for i in 0..20 {
            test_rsvd(100, 100, 10, 1, i, 1e-2);
        }
    }

    // test 100x10 matrix k=10
    #[test]
    fn test_rsvd_100x10_k10() {
        for i in 0..20 {
            test_rsvd(100, 10, 10, 1, i, 1e-2);
        }
    }

    // test 10x100 matrix k=10
    #[test]
    fn test_rsvd_10x100_k10() {
        for i in 0..20 {
            test_rsvd(10, 100, 10, 1, i, 1e-2);
        }
    }
}

#[cfg(test)]
mod pca_bench_tests {
    use super::*;
    use sysinfo::{System};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;
    use ndarray::Array2;

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
        let process_end = sys.process(pid).expect("Unable to get current process at end");
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
        println!("\n=== Test Case: {} ({} samples x {} features) ===",
                 scenario_name, n_samples, n_features);

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
        println!("----------+----------+----------+------------+------------+------------+------------");
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
