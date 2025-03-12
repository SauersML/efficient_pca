//! Principal component analysis (PCA)

#![doc = include_str!("../README.md")]

use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::svd::SVD;
use ndarray_linalg::eigh::Eigh;
use ndarray_linalg::UPLO;
use std::error::Error;
use ndarray_linalg::QR;
use rand::{thread_rng, Rng};
use rand_distr::Normal;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
    /// use pca::PCA;
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
    /// use pca::PCA;
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
        tolerance: Option<f64>
    ) -> Result<(), Box<dyn Error>> 
    {
        let num_samples = data_matrix.nrows();

        if num_samples < 2 {
            return Err("Input matrix must have at least 2 rows.".into());
        }

        // 1) Center and scale the data
        let mean_vector = data_matrix
            .mean_axis(Axis(0))
            .ok_or("Failed to compute mean of the data.")?;
        self.mean = Some(mean_vector.clone());
        data_matrix -= &mean_vector;

        let std_dev_vector = data_matrix.map_axis(Axis(0), |column| column.std(1.0));
        self.scale = Some(std_dev_vector.clone());
        data_matrix /= &std_dev_vector.mapv(|val| if val != 0.0 { val } else { 1.0 });

        // 2) Compute covariance-like matrix: shape (num_samples, num_samples)
        //    We do not necessarily divide by (num_samples - 1), because
        //    we only need the eigenvectors for principal directions.
        let cov_matrix = data_matrix.dot(&data_matrix.t());

        // 3) Eigen-decomposition of this symmetric (n×n) matrix
        let (eigenvalues, eigenvectors) = cov_matrix.eigh(UPLO::Upper)
            .map_err(|_| "Eigen decomposition of covariance matrix failed.")?;
        // shape of eigenvectors is (num_samples, num_samples)

        // 4) Sort eigenvalues and eigenvectors in descending order of eigenvalue
        //    (They may come in ascending or arbitrary order.)
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
            .iter()
            .cloned()
            .zip(eigenvectors.columns().into_iter().map(|col| col.to_owned()))
            .collect();

        // Sort descending by eigenvalue
        eigen_pairs.sort_by(|(val_a, _), (val_b, _)| val_b.partial_cmp(val_a).unwrap());

        // 5) Apply tolerance if provided (skip any eigenvalues below threshold of largest)
        //    For example, if tolerance=0.01, we exclude eigenvalues < 0.01 * largest_eigenvalue.
        let largest_eigenvalue = eigen_pairs.get(0).map_or(0.0, |(val, _)| *val);
        let rank_by_tolerance = if let Some(tol) = tolerance {
            let threshold = largest_eigenvalue * tol;
            eigen_pairs
                .iter()
                .take_while(|(val, _)| *val > threshold)
                .count()
        } else {
            eigen_pairs.len()
        };

        // 6) Build the top subset of eigenvectors
        //    We also can't exceed num_samples in rank.
        let final_rank = std::cmp::min(rank_by_tolerance, num_samples);
        let (top_eigenvalues, top_eigenvectors) = {
            let mut values = Vec::with_capacity(final_rank);
            let mut vectors = Vec::with_capacity(final_rank);

            for i in 0..final_rank {
                let (eval, evec) = &eigen_pairs[i];
                values.push(*eval);
                vectors.push(evec.clone());
            }
            // shape the vectors as (num_samples, final_rank)
            let stacked_vecs = ndarray::stack(
                Axis(1),
                &vectors.iter().map(|v| v.view()).collect::<Vec<_>>()
            )?;

            (values, stacked_vecs)
        };

        // 7) Map those eigenvectors back into feature space. 
        //    This recovers the principal axes in shape (num_features, final_rank).
        //    Formula: rotation = X^T * U / sqrt(eigenvalue) per column
        //    We'll do it by:
        //      (A) Multiply X^T * top_evecs, shape => (num_features, final_rank)
        //      (B) Optionally scale each column by 1/sqrt(eigenvalue)
        let mut rotation_matrix = data_matrix.t().dot(&top_eigenvectors);

        // 8) Optionally normalize each principal axis by sqrt of its eigenvalue,
        //    so that each axis is unit-norm in standard PCA sense.
        for (col_index, &eig_val) in top_eigenvalues.iter().enumerate() {
            // eigenvalue can be numerically near zero or negative (due to rounding),
            // so ensure safe sqrt:
            let scale_factor = if eig_val > 1e-12 {
                eig_val.sqrt()
            } else {
                1e-12
            };
            let mut column_mut = rotation_matrix.index_axis_mut(Axis(1), col_index);
            column_mut.mapv_inplace(|val| val / scale_factor);
        }

        self.rotation = Some(rotation_matrix);
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
    /// use pca::PCA;
    ///
    /// let x = array![[1.0, 2.0], [3.0, 4.0]];
    /// let mut pca = PCA::new();
    /// pca.rfit(x, 1, 0, None, None).unwrap();
    /// ```
    /// `rsvd` internally calls `.svd(true, true)`.
    pub fn rfit(&mut self, mut x: Array2<f64>,
                n_components: usize, n_oversamples: usize,
                seed: Option<u64>, tol: Option<f64>) -> Result<(), Box<dyn Error>> {
        let n = x.nrows();
        if n < 2 {
            return Err("Input matrix must have at least 2 rows.".into());
        }

        // Compute mean for centering
        let mean = x.mean_axis(Axis(0)).ok_or("Failed to compute mean")?;
        self.mean = Some(mean.clone());
        x -= &mean;

        // Compute scale
        let std_dev = x.map_axis(Axis(0), |v| v.std(1.0));
        self.scale = Some(std_dev.clone());
        x /= &std_dev.mapv(|v| if v != 0. { v } else { 1. });

        let k = std::cmp::min(n_components, std::cmp::min(n, x.ncols()));

        // Compute SVD
        let (_u, mut s, vt) = rsvd(&x, n_components, n_oversamples, seed);
        //.map_err(|_| "Failed to compute SVD")?;

        // Normalize singular values
        s.mapv_inplace(|v| v / ((n as f64 - 1.0).max(1.0)).sqrt());

        // Compute Rotation
        let rotation;
        if let Some(t) = tol {
            // convert diagonal matrix s into a vector
            let s = s.diag().to_owned();
            let threshold = s[0] * t;
            let rank = s.iter().take_while(|&si| *si > threshold).count();
            rotation = vt.slice_move(s![..std::cmp::min(rank, k), ..]).reversed_axes();
        } else {
            rotation = vt.slice_move(s![..k, ..]).reversed_axes();
        }
        self.rotation = Some(rotation);

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
    /// use pca::PCA;
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
pub fn rsvd(input: &Array2<f64>, k: usize, p: usize, seed: Option<u64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    //let m = input.shape()[0];
    let n = input.shape()[1];

    // handle the seed, which could be None
    // if it's None, we should use whatever default seeding is used by the RNG
    let rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_rng(thread_rng()).unwrap(),
    };

    // Generate Gaussian random test matrix
    let l = k + p; // Oversampling
    let omega = {
        let vec = rng.sample_iter(Normal::new(0.0, 1.0).unwrap())
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
    use ndarray::{Array2};
    use rand::Rng;
    use std::time::Instant;
    use sysinfo::{System};

    #[test]
    fn test_one_million_variants_88_haplotypes_binary() -> Result<(), Box<dyn std::error::Error>> {
        // Monitor memory usage before data generation
        let mut sys = System::new_all();
        sys.refresh_all();
        let initial_mem = sys.used_memory();
        println!("Initial memory usage: {} KB", initial_mem);

        // Dimensions: 88 haplotypes (rows) x 1,000,000 variants (columns)
        let n_rows = 88;
        let n_cols = 1_000_000;

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
        println!("Binary data generation completed in {:.2?}", data_gen_duration);

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
        assert_eq!(transformed.nrows(), n_rows, "Row count of transformed data is incorrect");
        assert_eq!(transformed.ncols(), n_components, "Column count of transformed data is incorrect");

        // Verify that none of the values are NaN or infinite
        for row in 0..n_rows {
            for col in 0..n_components {
                let val = transformed[[row, col]];
                assert!(val.is_finite(), "PCA output contains non-finite value");
            }
        }

        // Check memory usage afterwards
        sys.refresh_all();
        let final_mem = sys.used_memory();
        println!("Final memory usage: {} KB", final_mem);
        if final_mem < initial_mem {
            println!("Note: system-reported memory decreased; this can happen if other processes ended.");
        }

        println!("Test completed successfully with 1,000,000 variants x 88 haplotypes (binary).");
        Ok(())
    }
}

#[cfg(test)]
mod pca_tests {

    use ndarray::array;
    use ndarray_rand::rand_distr::Distribution;
    use super::*;
    use float_cmp::approx_eq;

    fn test_pca(input: Array2<f64>, expected_output: Array2<f64>, tol: Option<f64>, e: f64) {
        let mut pca = PCA::new();
        pca.fit(input.clone(), tol).unwrap();
        let output = pca.transform(input).unwrap();

        eprintln!("output: {:?}", output);
        eprintln!("expected_output: {:?}", expected_output);
        
        // Calculate absolute values for arrays
        let output_abs = output.mapv_into(f64::abs);
        let expected_output_abs = expected_output.mapv_into(f64::abs);

        // Compare arrays
        let equal = output_abs.shape() == expected_output_abs.shape() &&
            output_abs.iter().zip(expected_output_abs.iter())
                      .all(|(a, b)| approx_eq!(f64, *a, *b, epsilon = e));
        assert!(equal);
    }

    fn test_rpca(input: Array2<f64>, expected_output: Array2<f64>,
                 n_components: usize, n_oversamples: usize, tol: Option<f64>, e: f64) {
        let mut pca = PCA::new();
        pca.rfit(input.clone(), n_components, n_oversamples, Some(1926), tol).unwrap();
        let output = pca.transform(input).unwrap();

        eprintln!("output: {:?}", output);
        eprintln!("expected_output: {:?}", expected_output);
        
        // Calculate absolute values for arrays
        let output_abs = output.mapv_into(f64::abs);
        let expected_output_abs = expected_output.mapv_into(f64::abs);
        
        // Compare only up to the lesser of the two column counts.
        let min_cols = std::cmp::min(output_abs.ncols(), expected_output_abs.ncols());
        let output_slice = output_abs.slice(s![.., ..min_cols]);
        let expected_slice = expected_output_abs.slice(s![.., ..min_cols]);
        
        let equal = output_slice.iter().zip(expected_slice.iter())
            .all(|(a, b)| approx_eq!(f64, *a, *b, epsilon = e));
        assert!(equal);

    }

    #[test]
    fn test_pca_2x2() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_pca(input, expected, None, 1e-6);
    }

    #[test]
    fn test_rpca_2x2() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_rpca(input, expected, 2, 0, None, 1e-6);
    }

    #[test]
    fn test_rpca_2x2_k1() {
        let input = array![[0.5855288, -0.1093033], 
                           [0.7094660, -0.4534972]];
        let expected = array![[-1.0, 4.109492e-16],
                              [1.0, 2.647088e-16]];

        test_rpca(input, expected, 1, 0, None, 1e-6);
    }

    #[test]
    fn test_pca_3x5() {
        let input = array![[0.5855288, -0.4534972, 0.6300986, -0.9193220, 0.3706279],
                           [0.7094660, 0.6058875, -0.2761841, -0.1162478, 0.5202165],
                           [-0.1093033, -1.8179560, -0.2841597, 1.8173120, -0.7505320]];
        
        let expected = array![[-1.197063, 1.00708746, -2.503967e-17],
                              [-1.098771, -1.03624808, 1.566910e-16],
                              [2.295834, 0.02916061, 3.427952e-17]];

        test_pca(input, expected, None, 1e-6);
    }

    #[test]  
    fn test_pca_5x5() {
        let input = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219],
                           [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851],
                           [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284],
                           [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374],
                           [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095]];

        let expected = array![[0.7982358, 1.5873362, -0.81051708, 0.1302931, -1.372149e-16],
                              [-2.8355417, 0.1286742, -0.04173705, -0.2169158, -1.150065e-16],
                              [-0.1933991, -0.7382029, 0.34873461, 0.6235378, 8.618531e-17],
                              [1.0944225, -1.5788366, -0.57193457, -0.2771233, 4.276631e-16],
                              [1.1362824, 0.6010291, 1.07545409, -0.2597919, -6.793015e-17]];

        test_pca(input, expected, None, 1e-6);
    }

    #[test]
    fn test_pca_5x7() {
        let input = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
                           [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
                           [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
                           [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
                           [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]];
        
        let expected = array![[1.7585642, -1.3627442, -1.26793991, 0.050491148, -2.437707e-16],
                              [-3.0789816, -0.8169344, 0.05331594, 0.277390220, 3.028250e-15],
                              [-0.5847649, 0.8009802, -0.25561556, -0.836326771, -5.830345e-15],
                              [0.5487077, 1.8825193, -0.30961285, 0.511856077, 3.404246e-15],
                              [1.3564746, -0.5038210, 1.77985239, -0.003410674, -3.187086e-16]];

        test_pca(input, expected, None, 1e-6);
    }

    #[test]
    fn test_rpca_5x7_k3() {
        let input = array![[0.5855288, -1.8179560, -0.1162478, 0.8168998, 0.7796219, 1.8050975, 0.8118732],
                           [0.7094660, 0.6300986, 1.8173120, -0.8863575, 1.4557851, -0.4816474, 2.1968335],
                           [-0.1093033, -0.2761841, 0.3706279, -0.3315776, -0.6443284, 0.6203798, 2.0491903],
                           [-0.4534972, -0.2841597, 0.5202165, 1.1207127, -1.5531374, 0.6121235, 1.6324456],
                           [0.6058875, -0.9193220, -0.7505320, 0.2987237, -1.5977095, -0.1623110, 0.2542712]];
        
        let expected = array![[1.7585642, -1.3627442, -1.26793991, 0.050491148, -2.437707e-16],
                              [-3.0789816, -0.8169344, 0.05331594, 0.277390220, 3.028250e-15],
                              [-0.5847649, 0.8009802, -0.25561556, -0.836326771, -5.830345e-15],
                              [0.5487077, 1.8825193, -0.30961285, 0.511856077, 3.404246e-15],
                              [1.3564746, -0.5038210, 1.77985239, -0.003410674, -3.187086e-16]];

        test_rpca(input, expected, 4, 0, None, 1e-6);
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
        let input = Array2::<f64>::random_using((size, size), Uniform::new_inclusive(0, 2).map(|x| x as f64), &mut rng);

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
mod rsvd_tests {
    use super::*;
    use ndarray::Array2;

    // input a dimension, a seed, and a tolerance
    // we make a random matrix to match
    // then check if the rsvd is within a tolerance of the actual svd
    fn test_rsvd(m: usize, n:usize, k: usize, p: usize, seed: u64, tol: f64) {
        // Generate random matrix
        // use seeded RNG
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let a = Array2::from_shape_fn((m, n), |_| rng.gen::<f64>());

        // Compute rank approximation
        let (u, s, vt) = rsvd(&a, k, p, Some(1337));

        let (Some(u2), s2, Some(vt2)) = a.svd(true, true).unwrap()
        else { panic!("SVD failed"); };

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
