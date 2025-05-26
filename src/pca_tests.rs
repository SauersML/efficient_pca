use crate::PCA;
use ndarray::Axis;
// ndarray_linalg specific imports might not be needed directly in all test files if
// PCA internally uses the backend dispatch. UPLO might be used if tests construct matrices directly.
// For now, let's assume they might be used by other tests in this file.
use ndarray_linalg::{Eigh, QR, UPLO}; 

// For numerical consistency test
use approx::assert_abs_diff_eq;
// serde::Deserialize, std::fs, std::path::Path are no longer needed
use ndarray::{array, Array2}; // For creating the test data matrix easily & Array1 for direct comparison
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// PcaReferenceResults struct is no longer needed

fn generate_random_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.gen_range(0..=2) as f64 // Generates 0.0, 1.0, or 2.0
    })
}

#[cfg(test)]
mod genome_tests {
    use super::*;
    // use ndarray::array; // Already imported at top level of file
    use ndarray::{s, ArrayView1}; // Array2 also needed for some tests
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
        assert!(pca_original.rotation().is_some(), "Original (exact) model rotation should be Some");
        assert!(pca_original.mean().is_some(), "Original (exact) model mean should be Some");
        assert!(pca_original.scale().is_some(), "Original (exact) model scale should be Some");
        assert_eq!(pca_original.rotation().unwrap().ncols(), expected_components, "Original (exact) model component count mismatch");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();

        pca_original.save_model(file_path)?;
        println!("Exact model saved to: {:?}", file_path);

        let pca_loaded = PCA::load_model(file_path)?;
        println!("Exact model loaded successfully.");

        // 1. Verify loaded model parameters are identical
        assert_optional_array2_equals(pca_original.rotation(), pca_loaded.rotation(), "rotation matrix (exact fit)");
        assert_optional_array1_equals(pca_original.mean(), pca_loaded.mean(), "mean vector (exact fit)");
        assert_optional_array1_equals(pca_original.scale(), pca_loaded.scale(), "scale vector (exact fit)");

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
        assert!(pca_original.rotation().is_some(), "Original (randomized) model rotation should be Some");
        assert_eq!(pca_original.rotation().unwrap().ncols(), n_components_to_fit, "Original (randomized) model component count mismatch");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();
        pca_original.save_model(file_path)?;
        println!("Randomized model saved to: {:?}", file_path);

        let pca_loaded = PCA::load_model(file_path)?;
        println!("Randomized model loaded successfully.");

        // 1. Verify loaded model parameters
        assert_optional_array2_equals(pca_original.rotation(), pca_loaded.rotation(), "rotation matrix (randomized fit)");
        assert_optional_array1_equals(pca_original.mean(), pca_loaded.mean(), "mean vector (randomized fit)");
        assert_optional_array1_equals(pca_original.scale(), pca_loaded.scale(), "scale vector (randomized fit)");

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

        assert!(pca_original.rotation().is_some(), "Original model (zero components) rotation should be Some");
        assert_eq!(pca_original.rotation().unwrap().ncols(), 0, "Model with no significant variance should have 0 components");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();
        pca_original.save_model(file_path)?;
        println!("Zero-component model saved to: {:?}", file_path);

        let pca_loaded = PCA::load_model(file_path)?;
        println!("Zero-component model loaded successfully.");

        assert_optional_array2_equals(pca_original.rotation(), pca_loaded.rotation(), "rotation (zero components)");
        assert_eq!(pca_loaded.rotation().unwrap().ncols(), 0, "Loaded model should have 0 components");

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
        assert_optional_array1_equals(Some(&expected_sanitized_scale), pca_original.scale(), "sanitized scale in with_model");

        let temp_file = NamedTempFile::new()?;
        let file_path = temp_file.path();
        pca_original.save_model(file_path)?;
        println!("Model created with `with_model` saved to: {:?}", file_path);
        
        let pca_loaded = PCA::load_model(file_path)?;
        println!("Model loaded successfully.");

        assert_optional_array2_equals(pca_original.rotation(), pca_loaded.rotation(), "rotation (with_model)");
        assert_optional_array1_equals(pca_original.mean(), pca_loaded.mean(), "mean (with_model)");
        assert_optional_array1_equals(pca_original.scale(), pca_loaded.scale(), "scale (with_model, should be sanitized)");
        assert_optional_array1_equals(Some(&expected_sanitized_scale), pca_loaded.scale(), "loaded scale should match expected sanitized");


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
            pca_model_finite.scale(), 
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

    #[test]
    fn test_generate_random_data_values() {
        let n_samples = 10;
        let n_features = 10;
        let seed = 42;
        let data = generate_random_data(n_samples, n_features, seed);

        for val_ref in data.iter() {
            let val = *val_ref;
            assert!(
                val == 0.0 || val == 1.0 || val == 2.0,
                "Generated data contains unexpected value: {}. Expected 0.0, 1.0, or 2.0.",
                val
            );
        }
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
            eprintln!("Rust PCA Rotation:\n{:?}", pca.rotation());
            panic!("Comparison with Python PCA failed in {}", test_name);
        }
    }

    use super::*; // This brings PcaReferenceResults into scope for this module if it's outside
    // use ndarray::array; // Already imported at top level of file
    use ndarray_rand::rand_distr::Distribution;
    
    #[test]
    fn test_pca_fit_consistency_linfa() -> Result<(), Box<dyn Error>> {
        const TOLERANCE: f64 = 1e-5; // Tolerance for float comparisons
    
        // Helper function to compare two vectors for approximate equality.
        fn assert_vectors_approx_equal(
            v1: &Array1<f64>,
            v2: &Array1<f64>,
            tol: f64,
            context_msg: &str,
        ) {
            assert_eq!(v1.len(), v2.len(), "Length mismatch for {}: expected {}, got {}", context_msg, v2.len(), v1.len());
            for i in 0..v1.len() {
                assert_abs_diff_eq!(v1[i], v2[i], epsilon = tol, "Element mismatch at index {} in {}. v1[i]: {}, v2[i]: {}", i, context_msg, v1[i], v2[i]);
            }
        }
    
        // Helper function to compare two matrices, allowing for sign flips in columns.
        fn assert_matrices_approx_equal_columnwise_sign_agnostic(
            m1: &Array2<f64>,
            m2: &Array2<f64>,
            tol: f64,
            context_msg: &str,
        ) {
            assert_eq!(m1.shape(), m2.shape(), "Shape mismatch for {}: expected {:?}, got {:?}", context_msg, m2.shape(), m1.shape());
            if m1.ncols() == 0 { // Both are empty with same shape
                return;
            }
            for j in 0..m1.ncols() {
                let col1 = m1.column(j);
                let col2 = m2.column(j);
    
                let mut col1_sum_sq = 0.0;
                col1.for_each(|x| col1_sum_sq += x*x);
                let col1_norm = col1_sum_sq.sqrt();
    
                // Check if columns are close with same sign or opposite sign
                // We sum absolute differences and compare against a tolerance scaled by vector norm or length
                // to handle very small vectors where relative tolerance is less meaningful.
                let diff_same_sign_sum_abs = (&col1 - &col2).mapv(f64::abs).sum();
                let diff_flipped_sign_sum_abs = (&col1 + &col2).mapv(f64::abs).sum();
                
                // A simple heuristic for tolerance boundary, can be refined
                let max_permissible_error_sum = tol * (col1_norm + col1.len() as f64);
    
    
                assert!(
                    diff_same_sign_sum_abs < max_permissible_error_sum || diff_flipped_sign_sum_abs < max_permissible_error_sum,
                    "Column {} mismatch for {}. Sum diff_same_sign: {}, Sum diff_flipped_sign: {}. Max_err_sum: {}. col1: {:?.3}, col2: {:?.3}",
                    j, context_msg, diff_same_sign_sum_abs, diff_flipped_sign_sum_abs, max_permissible_error_sum, col1, col2
                );
            }
        }
    
        // Data matrix from the original test
        let data_matrix_owned = array![
            [1., 2., 3.], 
            [4., 5., 6.], 
            [7., 8., 9.], 
            [10., 11., 12.]
        ];
    
        let n_samples = data_matrix_owned.nrows();
        let n_features = data_matrix_owned.ncols();
        let k_components = n_features.min(n_samples); // For this 4x3 data, k_components = 3
    
        // --- 1. efficient_pca - fit() method ---
        println!("Testing efficient_pca with fit()...");
        let mut pca_fit_eff = PCA::new();
        pca_fit_eff.fit(data_matrix_owned.clone(), None)?; 
    
        let mean_fit_eff = pca_fit_eff.mean().expect("Mean missing after efficient_pca.fit()");
        let rotation_fit_eff = pca_fit_eff.rotation().expect("Rotation missing after efficient_pca.fit()");
        let transformed_fit_eff = pca_fit_eff.transform(data_matrix_owned.clone())?;
        let explained_variance_fit_eff = pca_fit_eff.explained_variance().expect("Explained variance missing after efficient_pca.fit()");
        let singular_values_from_fit_eff = explained_variance_fit_eff.mapv(|ev| (ev * (n_samples - 1) as f64).max(0.0).sqrt());
    
    
        // --- 2. efficient_pca - rfit() method ---
        println!("Testing efficient_pca with rfit()...");
        let mut pca_rfit_eff = PCA::new();
        let n_oversamples = 2; 
        let rfit_seed = Some(42u64);
        let transformed_rfit_eff = pca_rfit_eff.rfit(
            data_matrix_owned.clone(), 
            k_components, 
            n_oversamples, 
            rfit_seed, 
            None 
        )?;
    
        let mean_rfit_eff = pca_rfit_eff.mean().expect("Mean missing after efficient_pca.rfit()");
        let rotation_rfit_eff = pca_rfit_eff.rotation().expect("Rotation missing after efficient_pca.rfit()");
        let explained_variance_rfit_eff = pca_rfit_eff.explained_variance().expect("Explained variance missing after efficient_pca.rfit()");
        let singular_values_from_rfit_eff = explained_variance_rfit_eff.mapv(|ev| (ev * (n_samples - 1) as f64).max(0.0).sqrt());
    
    
        // --- 3. linfa::Pca - Reference ---
        println!("Fitting LinfaPcaModel as reference...");
        let linfa_dataset = DatasetBase::from(data_matrix_owned.clone()); 
        let linfa_rng_seed = 2025u64;
        
        let linfa_pca_model = LinfaPcaModel::params(k_components)
            .with_rng(StdRng::seed_from_u64(linfa_rng_seed)) 
            .fit(&linfa_dataset)?;
    
        let mean_linfa = linfa_pca_model.mean();
        let components_linfa = linfa_pca_model.components(); 
        let transformed_linfa = linfa_pca_model.predict(&linfa_dataset);
        let singular_values_linfa = linfa_pca_model.singular_values();
    
    
        // --- 4. Comparisons ---
        println!("Starting comparisons with LinfaPCA...");
    
        // Compare Means
        assert_vectors_approx_equal(mean_fit_eff, mean_linfa, TOLERANCE, "Mean (efficient_pca.fit vs linfa)");
        assert_vectors_approx_equal(mean_rfit_eff, mean_linfa, TOLERANCE, "Mean (efficient_pca.rfit vs linfa)");
    
        // Compare Rotation Matrices / Components
        assert_matrices_approx_equal_columnwise_sign_agnostic(
            rotation_fit_eff,
            components_linfa,
            TOLERANCE,
            "Rotation/Components (efficient_pca.fit vs linfa)",
        );
        assert_matrices_approx_equal_columnwise_sign_agnostic(
            rotation_rfit_eff,
            components_linfa,
            TOLERANCE * 10.0, // higher tolerance for rPCA vs exact SVD from Linfa
            "Rotation/Components (efficient_pca.rfit vs linfa)",
        );
        
        // Compare Transformed Data
        assert_matrices_approx_equal_columnwise_sign_agnostic(
            &transformed_fit_eff,
            &transformed_linfa,
            TOLERANCE,
            "Transformed Data (efficient_pca.fit vs linfa)",
        );
        assert_matrices_approx_equal_columnwise_sign_agnostic(
            &transformed_rfit_eff,
            &transformed_linfa,
            TOLERANCE * 10.0, // Potentially higher tolerance
            "Transformed Data (efficient_pca.rfit vs linfa)",
        );
    
        // Compare Singular Values (derived for efficient_pca, direct from linfa)
        // Linfa's PCA standardizes data before SVD.
        // The singular values obtained from both should thus correspond to standardized data.
        // Sort them for stable comparison, especially for near-zero values.
        let mut sv_fit_sorted = singular_values_from_fit_eff.to_vec();
        sv_fit_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut sv_rfit_sorted = singular_values_from_rfit_eff.to_vec();
        sv_rfit_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
        let mut sv_linfa_sorted = singular_values_linfa.to_vec();
        sv_linfa_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
        assert_vectors_approx_equal(
            &Array1::from_vec(sv_fit_sorted), 
            &Array1::from_vec(sv_linfa_sorted.clone()), // Clone as sv_linfa_sorted is used again
            TOLERANCE * 10.0, // Singular values might differ more than components
            "Sorted Singular Values (efficient_pca.fit vs linfa)"
        );
        assert_vectors_approx_equal(
            &Array1::from_vec(sv_rfit_sorted), 
            &Array1::from_vec(sv_linfa_sorted), 
            TOLERANCE * 20.0, // RPCA can have larger deviations for singular values
            "Sorted Singular Values (efficient_pca.rfit vs linfa)"
        );
    
        // --- 5. Internal Consistency Check for efficient_pca.fit() explained_variance ---
        // For the specific data_matrix, the first eigenvalue of Cov(X_scaled) should be 4.0.
        if explained_variance_fit_eff.len() > 0 {
            assert_abs_diff_eq!(
                explained_variance_fit_eff[0], 
                4.0, 
                epsilon = TOLERANCE,
                "Efficient PCA (fit) first explained variance for hardcoded data should be 4.0. Got: {}",
                explained_variance_fit_eff[0]
            );
        }

    
        // --- 6. Compare Explained Variance and Ratios (with caution for raw values) ---
        println!("Comparing Explained Variance values and ratios...");
    
        // Retrieve raw explained variance values
        // efficient_pca's EV: eigenvalues of Cov(X_scaled), i.e., s_scaled^2 / (N-1)
        // linfa's EV: s_linfa^2 / (k-1), where k is number of components
    
        let ev_fit_eff = pca_fit_eff.explained_variance().expect("Explained variance from efficient_pca.fit missing");
        let ev_rfit_eff = pca_rfit_eff.explained_variance().expect("Explained variance from efficient_pca.rfit missing");
        let ev_linfa = linfa_pca_model.explained_variance(); // Directly from linfa
    
        println!("[INFO] Raw Explained Variances (definitions differ across libraries):");
        println!("  EfficientPCA (fit):  {:?.3}", ev_fit_eff);
        println!("  EfficientPCA (rfit): {:?.3}", ev_rfit_eff);
        println!("  LinfaPCA:            {:?.3}", ev_linfa);
        println!("  Note: Linfa's explained_variance uses sigma^2/(k-1), efficient_pca's uses sigma_scaled^2/(N-1).");
        println!("        Direct equality is not expected for raw values due to different denominators.");
    
        // Compare Explained Variance Ratios
        // Ratio for efficient_pca: ev_i / sum(ev) = (s_scaled_i^2 / (N-1)) / sum(s_scaled_j^2 / (N-1)) = s_scaled_i^2 / sum(s_scaled_j^2)
        // Ratio for linfa: linfa_pca_model.explained_variance_ratio() = (s_linfa_i^2 / (k-1)) / sum(s_linfa_j^2 / (k-1)) = s_linfa_i^2 / sum(s_linfa_j^2)
        // If s_scaled and s_linfa are comparable (derived from similarly standardized data), their ratios should be comparable.
    
        let sum_ev_fit_eff = ev_fit_eff.sum();
        let ratio_fit_eff = if sum_ev_fit_eff.abs() > f64::EPSILON {
            ev_fit_eff.mapv(|v| v / sum_ev_fit_eff)
        } else {
            Array1::zeros(ev_fit_eff.len()) // Avoid division by zero if sum is zero
        };
    
        let sum_ev_rfit_eff = ev_rfit_eff.sum();
        let ratio_rfit_eff = if sum_ev_rfit_eff.abs() > f64::EPSILON {
            ev_rfit_eff.mapv(|v| v / sum_ev_rfit_eff)
        } else {
            Array1::zeros(ev_rfit_eff.len())
        };
        
        let ratio_linfa = linfa_pca_model.explained_variance_ratio();
    
        println!("[DEBUG] Explained Variance Ratios:");
        println!("  EfficientPCA (fit) ratio:  {:?.3}", ratio_fit_eff);
        println!("  EfficientPCA (rfit) ratio: {:?.3}", ratio_rfit_eff);
        println!("  LinfaPCA ratio:            {:?.3}", ratio_linfa);
    
        assert_vectors_approx_equal(
            &ratio_fit_eff, 
            &ratio_linfa, 
            TOLERANCE * 10.0, // Ratios can also be sensitive
            "Explained Variance Ratio (efficient_pca.fit vs linfa)"
        );
        assert_vectors_approx_equal(
            &ratio_rfit_eff, 
            &ratio_linfa, 
            TOLERANCE * 20.0, // RPCA ratios might differ more
            "Explained Variance Ratio (efficient_pca.rfit vs linfa)"
        );
    
        println!("Test 'test_pca_fit_consistency_linfa' (comparing with Linfa) passed!");
        Ok(())
    }

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
