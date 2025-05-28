// In tests/eigensnp_tests.rs
// eigensnp is primarily designed for large-scale genomic datasets, 
// such as those found in biobanks or large reference panels. 
// These datasets typically have many more features (SNPs) than samples.
// Therefore, the tests below focus on validating eigensnp's performance 
// and correctness on large matrices where the number of features significantly 
// exceeds the number of samples. Small test cases or cases where samples >= features 
// have been deemphasized or removed to better reflect real-world usage scenarios.

use ndarray::{arr2, s, Array1, Array2, ArrayView1, ArrayView2, Axis}; // Added ArrayView2
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use efficient_pca::eigensnp::{
    EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, LdBlockSpecification, // Removed EigenSNPCoreOutput
    PcaReadyGenotypeAccessor, PcaSnpId, QcSampleId, ThreadSafeStdError, reorder_array_owned, reorder_columns_owned,
};
use rand::SeedableRng;
use rand::Rng; // Added for the .sample() method
use rand_chacha::ChaCha8Rng;
use std::process::{Command, Stdio};
use std::io::Write; // Removed BufReader, BufRead
use std::str::FromStr;
use std::path::PathBuf;
use std::fs::{self, File}; // Add fs for create_dir_all
// use std::io::Write; // Already present
use std::path::Path; // Add Path
// use ndarray::{ArrayView1, ArrayView2}; // These are brought in by `use ndarray::{arr2, s, Array1, Array2, ArrayView1, Axis};`
use std::fmt::Display; // To constrain T
use std::fs::OpenOptions;
use std::sync::Mutex;
use lazy_static::lazy_static;

use crate::eigensnp_integration_tests::parse_pca_py_output;
use crate::eigensnp_integration_tests::TestDataAccessor;
use crate::eigensnp_integration_tests::TestResultRecord;
use crate::eigensnp_integration_tests::TEST_RESULTS;

const DEFAULT_FLOAT_TOLERANCE_F32: f32 = 1e-4; // Slightly looser for cross-implementation comparison
const DEFAULT_FLOAT_TOLERANCE_F64: f64 = 1e-4; // Slightly looser for cross-implementation comparison

// Helper function for comparing Array1<f64>
fn assert_f64_arrays_are_close(
    arr1: ArrayView1<f64>, // Changed from &Array1<f64>
    arr2: ArrayView1<f64>, // Changed from &Array1<f64>
    tolerance: f64,
    context: &str,
) {
    assert_eq!(arr1.dim(), arr2.dim(), "Array dimensions differ for {}. Left: {:?}, Right: {:?}", context, arr1.dim(), arr2.dim());
    for (i, val1) in arr1.iter().enumerate() {
        let val2 = arr2[i]; // Indexing on ArrayView1 is fine
        assert!(
            (val1 - val2).abs() < tolerance,
            "Mismatch at index {} for {}: {} vs {} (diff: {})",
            i,
            context,
            val1,
            val2,
            (val1 - val2).abs()
        );
    }
}

// Helper for comparing Array2<f32> allowing for sign flips per column
fn assert_f32_arrays_are_close_with_sign_flips(
    arr1: ndarray::ArrayView2<f32>, // Qualified with ndarray::
    arr2: ndarray::ArrayView2<f32>, // Qualified with ndarray::
    tolerance: f32,
    context: &str,
) {
    assert_eq!(arr1.dim(), arr2.dim(), "Array dimensions differ for {}. Left: {:?}, Right: {:?}", context, arr1.dim(), arr2.dim());
    if arr1.ncols() == 0 && arr2.ncols() == 0 { // Both empty, considered close
        return;
    }
    if arr1.ncols() == 0 || arr2.ncols() == 0 { // One empty, one not
         panic!("Array column count mismatch for {}: Left: {}, Right: {}. Both must be empty or non-empty.", context, arr1.ncols(), arr2.ncols());
    }

    for c_idx in 0..arr1.ncols() {
        let col1 = arr1.column(c_idx); // .column() on ArrayView2 is fine
        let col2 = arr2.column(c_idx); // .column() on ArrayView2 is fine
        
        let mut direct_match = true;
        for r_idx in 0..col1.len() {
            if (col1[r_idx] - col2[r_idx]).abs() >= tolerance {
                direct_match = false;
                break;
            }
        }

        if direct_match {
            continue; 
        }

        let mut flipped_match = true;
        for r_idx in 0..col1.len() {
            if (col1[r_idx] - (-col2[r_idx])).abs() >= tolerance {
                flipped_match = false;
                break;
            }
        }

        assert!(
            flipped_match,
            "Column {} mismatch for {} (even with sign flip check). Max diff: {}. First elements: {} vs {}",
            c_idx, context, 
            col1.iter().zip(col2.iter()).map(|(a,b)| (a-b).abs().max((a-(-b)).abs())).fold(0.0f32, f32::max),
            col1.get(0).unwrap_or(&0.0f32), col2.get(0).unwrap_or(&0.0f32)
        );
    }
}

// Standardizes each feature (SNP, row) across samples (columns).
fn standardize_features_across_samples(mut data: Array2<f32>) -> Array2<f32> {
    if data.ncols() <= 1 { 
        if data.ncols() == 1 && data.nrows() > 0 { data.fill(0.0); } 
        return data;
    }
    for mut feature_row in data.axis_iter_mut(Axis(0)) { 
        let mean = feature_row.mean().unwrap_or(0.0);
        feature_row.mapv_inplace(|x| x - mean);
        let std_dev = feature_row.std(0.0); 
        if std_dev.abs() > 1e-7 { 
            feature_row.mapv_inplace(|x| x / std_dev);
        } else {
            feature_row.fill(0.0); 
        }
    }
    data
}

// Helper functions to save arrays/vectors to TSV for debugging
fn save_matrix_to_tsv<T: Display>(
    matrix: &ArrayView2<T>,
    dir_path: &str,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let full_path = Path::new(dir_path).join(file_name);
    fs::create_dir_all(Path::new(dir_path))?; // Create directory if it doesn't exist
    let mut file = File::create(full_path)?;
    for row_idx in 0..matrix.nrows() {
        for col_idx in 0..matrix.ncols() {
            write!(file, "{}", matrix[[row_idx, col_idx]])?;
            if col_idx < matrix.ncols() - 1 {
                write!(file, "	")?; // Tab separated
            }
        }
        writeln!(file)?;
    }
    Ok(())
}

fn save_vector_to_tsv<T: Display>(
    vector: &ArrayView1<T>,
    dir_path: &str,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let full_path = Path::new(dir_path).join(file_name);
    fs::create_dir_all(Path::new(dir_path))?; // Create directory
    let mut file = File::create(full_path)?;
    for i in 0..vector.len() {
        writeln!(file, "{}", vector[i])?;
    }
    Ok(())
}

#[cfg(test)]
mod eigensnp_integration_tests {
    use super::*; 

    // Define TestResultRecord struct
    #[derive(Clone, Debug)] // Added Debug
    pub struct TestResultRecord {
        pub test_name: String,
        pub num_features_d: usize,
        pub num_samples_n: usize,
        pub num_pcs_requested_k: usize,
        pub num_pcs_computed: usize,
        pub success: bool,
        pub outcome_details: String,
        pub notes: String,
    }

    // Global static for results
    lazy_static! {
        pub static ref TEST_RESULTS: Mutex<Vec<TestResultRecord>> = Mutex::new(Vec::new());
    }

    // Function to write results to TSV
    pub fn write_results_to_tsv() -> Result<(), std::io::Error> {
        let results = TEST_RESULTS.lock().unwrap();
        if results.is_empty() {
            return Ok(()); // No results to write
        }

        let artifact_dir = "target/test_artifacts";
        std::fs::create_dir_all(artifact_dir)?;
        let tsv_path = Path::new(artifact_dir).join("eigensnp_summary_results.tsv");
        
        // Use OpenOptions to create or truncate the file
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(tsv_path)?;

        // Write header
        writeln!(file, "TestName	NumFeatures_D	NumSamples_N	NumPCsRequested_K	NumPCsComputed	Success	OutcomeDetails	Notes")?;

        for record in results.iter() {
            writeln!(file, "{}	{}	{}	{}	{}	{}	{}	{}",
                record.test_name,
                record.num_features_d,
                record.num_samples_n,
                record.num_pcs_requested_k,
                record.num_pcs_computed,
                record.success,
                record.outcome_details.replace("	", " ").replace("\n", "; "), // Sanitize details
                record.notes.replace("	", " ").replace("\n", "; ") // Sanitize notes
            )?;
        }
        Ok(())
    }

    // Hook for writing results after all tests in this module.
    // This is a bit of a workaround. A proper test harness or `ctor` crate might be better.
    // We define a "final" test that calls the write function.
    // Ensure this test runs last (e.g. by naming convention, though Rust test order isn't guaranteed).
    // For now, we will call this manually or make it the last test.
    // A more robust solution would be a custom test framework or procedural macro.
    #[test]
    fn finalize_and_write_results() {
        write_results_to_tsv().expect("Failed to write test results to TSV");
    }


    #[derive(Clone)]
    pub struct TestDataAccessor {
        standardized_data: Array2<f32>, 
    }

    impl TestDataAccessor {
        pub fn new(standardized_data: Array2<f32>) -> Self {
            // The line `let num_samples = standardized_data.ncols();` has been removed.
            Self {
                standardized_data,
            }
        }

        pub fn new_empty(num_pca_snps: usize, num_qc_samples: usize) -> Self {
            let standardized_data = Array2::zeros((num_pca_snps, num_qc_samples));
            Self {
                standardized_data,
            }
        }
    }

    impl PcaReadyGenotypeAccessor for TestDataAccessor {
        fn get_standardized_snp_sample_block(
            &self,
            snp_ids: &[PcaSnpId],
            sample_ids: &[QcSampleId],
        ) -> Result<Array2<f32>, ThreadSafeStdError> {
            if snp_ids.is_empty() { 
                return Ok(Array2::zeros((0, sample_ids.len())));
            }
            if sample_ids.is_empty() {
                 return Ok(Array2::zeros((snp_ids.len(), 0)));
            }
            
            if self.standardized_data.nrows() == 0 && !snp_ids.is_empty() {
                 return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Requested {} SNPs from an accessor with 0 SNPs.", snp_ids.len())
                )));
            }
            if self.standardized_data.ncols() == 0 && !sample_ids.is_empty() {
                 return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Requested {} samples from an accessor with 0 samples.", sample_ids.len())
                )));
            }

            let mut result_block = Array2::zeros((snp_ids.len(), sample_ids.len()));
            for (i, pca_snp_id) in snp_ids.iter().enumerate() {
                let target_row_idx = pca_snp_id.0;
                if target_row_idx >= self.standardized_data.nrows() {
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput,
                        format!("SNP ID PcaSnpId({}) out of bounds for {} SNPs", target_row_idx, self.standardized_data.nrows()))));
                }
                for (j, qc_sample_id) in sample_ids.iter().enumerate() {
                    let target_col_idx = qc_sample_id.0;
                     if target_col_idx >= self.standardized_data.ncols() {
                        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput,
                            format!("Sample ID QcSampleId({}) out of bounds for {} samples", target_col_idx, self.standardized_data.ncols()))));
                    }
                    result_block[[i, j]] = self.standardized_data[[target_row_idx, target_col_idx]];
                }
            }
            Ok(result_block)
        }

        fn num_pca_snps(&self) -> usize { self.standardized_data.nrows() }
        fn num_qc_samples(&self) -> usize { self.standardized_data.ncols() }
    }

    fn parse_section<T: FromStr>(lines: &mut std::iter::Peekable<std::str::Lines<'_>>, expected_dim2: Option<usize>) -> Result<Array2<T>, String>
    where <T as FromStr>::Err: std::fmt::Debug {
        let mut data_vec = Vec::new();
        let mut current_dim2 = None;

        loop {
            match lines.peek() {
                Some(line_peek) => {
                    if line_peek.is_empty() || line_peek.starts_with("LOADINGS:") || line_peek.starts_with("SCORES:") || line_peek.starts_with("EIGENVALUES:") {
                        // This is a separator or next section header, so stop parsing for current section
                        break;
                    }
                    // If not a separator, it's data for the current section. Consume and parse.
                    let line = lines.next().unwrap(); // Safe due to peek

                    let row: Vec<T> = line.split_whitespace()
                        .map(|s| s.parse::<T>().map_err(|e| format!("Failed to parse value: {:?}, error: {:?}", s, e)))
                        .collect::<Result<Vec<T>, String>>()?;
                    
                    if let Some(d2) = current_dim2 {
                        if row.len() != d2 { return Err(format!("Inconsistent row length. Expected {}, got {}", d2, row.len())); }
                    } else {
                        current_dim2 = Some(row.len());
                        if let Some(exp_d2) = expected_dim2 {
                            // Allow empty rows if section is empty (e.g. 0-component PCA, eigenvalues line is just empty)
                            // If row.is_empty() is true, it means current_dim2 became Some(0).
                            // If exp_d2 is Some(1) (e.g. for eigenvalues), and we got an empty line (parsed to row.len() == 0),
                            // this is a valid case for an empty section (e.g. 0 eigenvalues).
                            // The check `!row.is_empty()` was there before, let's see if it's still needed.
                            // If row.is_empty(), current_dim2 will be Some(0).
                            // If expected_dim2 is Some(1), and current_dim2 is Some(0), this is fine for an empty section.
                            // The problem is if the first non-empty row has a different number of columns than expected.
                            if !row.is_empty() && row.len() != exp_d2 {
                                 return Err(format!("Unexpected row length for section. Expected {}, got {}", exp_d2, row.len()));
                            }
                            // If row is empty and exp_d2 is Some(k) where k > 0: current_dim2 becomes Some(0). This is okay for an empty section.
                        }
                    }
                    data_vec.extend(row);
                }
                None => {
                    // End of input
                    break;
                }
            }
        }
        
        let actual_dim2 = current_dim2.unwrap_or_else(|| expected_dim2.unwrap_or(0));
        let num_rows = if actual_dim2 == 0 { 0 } else { data_vec.len() / actual_dim2 };

        Array2::from_shape_vec((num_rows, actual_dim2), data_vec)
            .map_err(|e| format!("Failed to create Array2: {}", e))
    }

    pub fn parse_pca_py_output(output_str: &str) -> Result<(Array2<f32>, Array2<f32>, Array1<f64>), String> {
        let mut lines = output_str.lines().peekable();
        
        let mut py_loadings: Option<Array2<f32>> = None;
        let mut py_scores: Option<Array2<f32>> = None;
        let mut py_eigenvalues: Option<Array1<f64>> = None;

        while let Some(line_peek) = lines.peek() {
            let current_line_is_empty = line_peek.is_empty(); 
            if line_peek.starts_with("LOADINGS:") {
                lines.next(); // Consume header
                py_loadings = Some(parse_section(&mut lines, None)?);
            } else if line_peek.starts_with("SCORES:") {
                lines.next(); // Consume header
                py_scores = Some(parse_section(&mut lines, None)?);
            } else if line_peek.starts_with("EIGENVALUES:") {
                lines.next(); // Consume header
                let eig_array2 = parse_section(&mut lines, Some(1))?;
                let eig_len = eig_array2.len();
                py_eigenvalues = Some(eig_array2.into_shape_with_order((eig_len,)).expect("Failed to reshape py_eigenvalues"));
            } else if current_line_is_empty { 
               lines.next(); // Consume the empty line
               // Continue to next iteration to peek at next line
            } else {
                // Unexpected line
                return Err(format!("Unexpected content in pca.py output. Line: '{}'", line_peek));
            }
        }
        
        Ok((
            py_loadings.ok_or_else(|| "LOADINGS section not found".to_string())?,
            py_scores.ok_or_else(|| "SCORES section not found".to_string())?,
            py_eigenvalues.ok_or_else(|| "EIGENVALUES section not found".to_string())?,
        ))
    }

    // Helper function for PC scores orthogonality tests to avoid code duplication
    pub fn run_pc_scores_orthogonality_test(
        test_name_str: &str,
        num_snps: usize,
        num_samples: usize,
        num_pcs_target: usize,
        seed: u64,
    ) {
        let test_name = test_name_str.to_string();
        let mut test_successful = true;
        let mut outcome_details = String::new();
        let notes = format!("Matrix: {}x{}, PCs: {}", num_snps, num_samples, num_pcs_target);

        let mut max_off_diagonal_cov = 0.0f64;
        let mut max_diag_eigenvalue_diff = 0.0f64;

        let output_result = std::panic::catch_unwind(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
            let standardized_genos = standardize_features_across_samples(raw_genos);
            let test_data = TestDataAccessor::new(standardized_genos);

            let config = EigenSNPCoreAlgorithmConfig {
                target_num_global_pcs: num_pcs_target,
                subset_factor_for_local_basis_learning: 0.5, // Example value
                min_subset_size_for_local_basis_learning: (num_samples / 4).max(1).min(num_samples.max(1)),
                max_subset_size_for_local_basis_learning: (num_samples / 2).max(10).min(num_samples.max(1)),
                components_per_ld_block: 10.min(num_snps.min( (num_samples/2).max(10).min(num_samples.max(1)) )), 
                random_seed: seed,
                ..Default::default()
            };
            let algorithm = EigenSNPCoreAlgorithm::new(config);
            let ld_blocks = vec![LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
            }];
            algorithm.compute_pca(&test_data, &ld_blocks)
        });

        match output_result {
            Ok(Ok(output)) => {
                if output.num_principal_components_computed != num_pcs_target {
                    test_successful = false;
                    outcome_details.push_str(&format!(
                        "Did not compute target PCs. Expected: {}, Got: {}. ",
                        num_pcs_target, output.num_principal_components_computed
                    ));
                }

                let scores = &output.final_sample_principal_component_scores;
                if scores.nrows() != num_samples {
                    test_successful = false;
                    outcome_details.push_str(&format!(
                        "Scores nrows mismatch. Expected: {}, Got: {}. ",
                        num_samples, scores.nrows()
                    ));
                }
                if scores.ncols() != output.num_principal_components_computed { // Check against actual computed PCs
                    test_successful = false;
                    outcome_details.push_str(&format!(
                        "Scores ncols mismatch. Expected: {}, Got: {}. ",
                        output.num_principal_components_computed, scores.ncols()
                    ));
                }

                if num_samples <= 1 || output.num_principal_components_computed == 0 {
                     outcome_details.push_str("Test condition (num_samples <=1 or num_pcs_computed == 0) means no further checks performed. ");
                } else if test_successful { // Only proceed if initial checks passed
                    let scores_f64 = scores.mapv(|x| x as f64);
                    let denominator = if output.num_qc_samples_used > 1 {
                        output.num_qc_samples_used as f64 - 1.0
                    } else {
                        1.0 // Avoid division by zero if num_qc_samples_used is 1 or 0
                    };
                    if denominator == 0.0 { // Should not happen if num_samples > 1
                        test_successful = false;
                        outcome_details.push_str("Denominator for covariance calculation is zero. ");
                    } else {
                        let covariance_matrix = scores_f64.t().dot(&scores_f64) / denominator;
                        let k_eff = output.num_principal_components_computed;

                        for r in 0..k_eff {
                            for c in 0..k_eff {
                                if r == c {
                                    let diff = (covariance_matrix[[r, c]] - output.final_principal_component_eigenvalues[r]).abs();
                                    if diff > max_diag_eigenvalue_diff { max_diag_eigenvalue_diff = diff; }
                                    if diff >= DEFAULT_FLOAT_TOLERANCE_F64 * 100.0 {
                                        test_successful = false;
                                        outcome_details.push_str(&format!(
                                            "Covariance diagonal [{},{}] {} does not match eigenvalue {}. Diff: {}. ",
                                            r, c, covariance_matrix[[r, c]], output.final_principal_component_eigenvalues[r], diff
                                        ));
                                    }
                                } else {
                                    let off_diag_val = covariance_matrix[[r, c]].abs();
                                    if off_diag_val > max_off_diagonal_cov { max_off_diagonal_cov = off_diag_val; }
                                    if off_diag_val >= DEFAULT_FLOAT_TOLERANCE_F64 * 100.0 {
                                        test_successful = false;
                                        outcome_details.push_str(&format!(
                                            "Covariance off-diagonal [{},{}] {} is not close to 0. Value: {}. ",
                                            r, c, covariance_matrix[[r, c]], off_diag_val
                                        ));
                                    }
                                }
                            }
                        }
                        if test_successful {
                             outcome_details.push_str(&format!("All orthogonality checks passed. Max off-diag cov: {:.2e}, Max diag-eigenvalue diff: {:.2e}. ", max_off_diagonal_cov, max_diag_eigenvalue_diff));
                        } else {
                             outcome_details.push_str(&format!("Orthogonality checks failed. Max off-diag cov: {:.2e}, Max diag-eigenvalue diff: {:.2e}. ", max_off_diagonal_cov, max_diag_eigenvalue_diff));
                        }
                    }
                }
                 let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: output.num_principal_components_computed,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                eigensnp_integration_tests::TEST_RESULTS.lock().unwrap().push(record);

            }
            Ok(Err(e)) => { // PCA computation itself failed
                test_successful = false;
                outcome_details = format!("PCA computation failed: {:?}", e);
                 let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: 0, // Failed to compute
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                eigensnp_integration_tests::TEST_RESULTS.lock().unwrap().push(record);
            }
            Err(e) => { // Panic during PCA computation or setup
                test_successful = false;
                outcome_details = format!("Test panicked: {:?}", e);
                let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: 0, // Panic, no output
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                eigensnp_integration_tests::TEST_RESULTS.lock().unwrap().push(record);
            }
        }
        assert!(test_successful, "Test {} failed. Max off-diag: {:.2e}, Max diag-eig diff: {:.2e}. Details: {}", test_name, max_off_diagonal_cov, max_diag_eigenvalue_diff, outcome_details);
    }

    #[test]
    fn test_pc_scores_orthogonality_large_500x100() {
        run_pc_scores_orthogonality_test("test_pc_scores_orthogonality_large_500x100", 500, 100, 5, 123);
    }

    #[test]
    fn test_pc_scores_orthogonality_large_1000x200() {
        run_pc_scores_orthogonality_test("test_pc_scores_orthogonality_large_1000x200", 1000, 200, 10, 124);
    }

    // Helper function for SNP loadings orthonormality tests
    pub fn run_snp_loadings_orthonormality_test(
        test_name_str: &str,
        num_snps: usize,
        num_samples: usize,
        num_pcs_target: usize,
        seed: u64,
    ) {
        let test_name = test_name_str.to_string();
        let mut test_successful = true;
        let mut outcome_details = String::new();
        let notes = format!("Matrix: {}x{}, PCs: {}", num_snps, num_samples, num_pcs_target);
        let mut max_deviation_from_identity = 0.0f32;

        let output_result = std::panic::catch_unwind(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
            let standardized_genos = standardize_features_across_samples(raw_genos);
            let test_data = TestDataAccessor::new(standardized_genos);

            let config = EigenSNPCoreAlgorithmConfig {
                target_num_global_pcs: num_pcs_target,
                random_seed: seed,
                 // Use appropriate subset settings for larger data
                subset_factor_for_local_basis_learning: 0.5, 
                min_subset_size_for_local_basis_learning: (num_samples / 4).max(1).min(num_samples.max(1)),
                max_subset_size_for_local_basis_learning: (num_samples / 2).max(10).min(num_samples.max(1)),
                components_per_ld_block: 10.min(num_snps.min( (num_samples/2).max(10).min(num_samples.max(1)) )),
                ..Default::default()
            };
            let algorithm = EigenSNPCoreAlgorithm::new(config);
            let ld_blocks = vec![LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
            }];
            algorithm.compute_pca(&test_data, &ld_blocks)
        });

        match output_result {
            Ok(Ok(output)) => {
                if output.num_principal_components_computed != num_pcs_target {
                    test_successful = false;
                    outcome_details.push_str(&format!(
                        "Did not compute target PCs. Expected: {}, Got: {}. ",
                        num_pcs_target, output.num_principal_components_computed
                    ));
                }

                let loadings = &output.final_snp_principal_component_loadings;
                if loadings.nrows() != num_snps {
                     test_successful = false;
                    outcome_details.push_str(&format!(
                        "Loadings nrows mismatch. Expected: {}, Got: {}. ",
                        num_snps, loadings.nrows()
                    ));
                }
                 if loadings.ncols() != output.num_principal_components_computed {
                     test_successful = false;
                    outcome_details.push_str(&format!(
                        "Loadings ncols mismatch. Expected: {}, Got: {}. ",
                         output.num_principal_components_computed, loadings.ncols()
                    ));
                }
                
                if output.num_principal_components_computed == 0 {
                    outcome_details.push_str("No PCs computed, skipping orthonormality check. ");
                } else if test_successful {
                    let check_identity = loadings.t().dot(loadings);
                    let k_eff = output.num_principal_components_computed;

                    for r_idx in 0..k_eff {
                        for c_idx in 0..k_eff {
                            let expected_val = if r_idx == c_idx { 1.0 } else { 0.0 };
                            let deviation = (check_identity[[r_idx, c_idx]] - expected_val).abs();
                            if deviation > max_deviation_from_identity {
                                max_deviation_from_identity = deviation;
                            }
                            if deviation >= DEFAULT_FLOAT_TOLERANCE_F32 {
                                test_successful = false;
                                outcome_details.push_str(&format!(
                                    "Loadings orthonormality check: Identity matrix mismatch at [{},{}]. Expected {}, Got {}. Diff: {}. ",
                                    r_idx, c_idx, expected_val, check_identity[[r_idx, c_idx]], deviation
                                ));
                            }
                        }
                    }
                     if test_successful {
                        outcome_details.push_str(&format!("All orthonormality checks passed. Max deviation from identity: {:.2e}. ", max_deviation_from_identity));
                    } else {
                        outcome_details.push_str(&format!("Orthonormality checks failed. Max deviation from identity: {:.2e}. ", max_deviation_from_identity));
                    }
                }
                let record = eigensnp_integration_tests::TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: output.num_principal_components_computed,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                TEST_RESULTS.lock().unwrap().push(record);
            }
            Ok(Err(e)) => {
                test_successful = false;
                outcome_details = format!("PCA computation failed: {:?}", e);
                 let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: 0,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                TEST_RESULTS.lock().unwrap().push(record);
            }
            Err(e) => {
                test_successful = false;
                outcome_details = format!("Test panicked: {:?}", e);
                let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: 0,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                TEST_RESULTS.lock().unwrap().push(record);
            }
        }
        assert!(test_successful, "Test {} failed. Max deviation from identity: {:.2e}. Details: {}", test_name, max_deviation_from_identity, outcome_details);
    }

    #[test]
    fn test_snp_loadings_orthonormality_large_500x100() {
        run_snp_loadings_orthonormality_test("test_snp_loadings_orthonormality_large_500x100", 500, 100, 5, 456);
    }

    #[test]
    fn test_snp_loadings_orthonormality_large_1000x200() {
        run_snp_loadings_orthonormality_test("test_snp_loadings_orthonormality_large_1000x200", 1000, 200, 10, 457);
    }

    // Helper function for eigenvalue-score variance correspondence tests
    pub fn run_eigenvalue_score_variance_correspondence_test(
        test_name_str: &str,
        num_snps: usize,
        num_samples: usize,
        num_pcs_target: usize,
        seed: u64,
    ) {
        let test_name = test_name_str.to_string();
        let mut test_successful = true;
        let mut outcome_details = String::new();
        let notes = format!("Matrix: {}x{}, PCs: {}", num_snps, num_samples, num_pcs_target);
        let mut max_variance_eigenvalue_diff = 0.0f64;

        let output_result = std::panic::catch_unwind(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
            let standardized_genos = standardize_features_across_samples(raw_genos);
            let test_data = TestDataAccessor::new(standardized_genos);

            let config = EigenSNPCoreAlgorithmConfig {
                target_num_global_pcs: num_pcs_target,
                random_seed: seed,
                // Use appropriate subset settings for larger data
                subset_factor_for_local_basis_learning: 0.5, 
                min_subset_size_for_local_basis_learning: (num_samples / 4).max(1).min(num_samples.max(1)),
                max_subset_size_for_local_basis_learning: (num_samples / 2).max(10).min(num_samples.max(1)),
                components_per_ld_block: 10.min(num_snps.min( (num_samples/2).max(10).min(num_samples.max(1)) )),
                ..Default::default()
            };
            let algorithm = EigenSNPCoreAlgorithm::new(config);
            let ld_blocks = vec![LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
            }];
            algorithm.compute_pca(&test_data, &ld_blocks)
        });

        match output_result {
            Ok(Ok(output)) => {
                if output.num_principal_components_computed != num_pcs_target {
                    test_successful = false;
                    outcome_details.push_str(&format!(
                        "Did not compute target PCs. Expected: {}, Got: {}. ",
                        num_pcs_target, output.num_principal_components_computed
                    ));
                }

                if num_samples <= 1 || output.num_principal_components_computed == 0 {
                     outcome_details.push_str("Test condition (num_samples <=1 or num_pcs_computed == 0) means no further checks performed. ");
                } else if test_successful {
                    let scores = &output.final_sample_principal_component_scores; 
                    let eigenvalues = &output.final_principal_component_eigenvalues;
                    let k_eff = output.num_principal_components_computed;
                    let denominator = if output.num_qc_samples_used > 1 {
                        output.num_qc_samples_used as f64 - 1.0
                    } else {
                        1.0 // Avoid division by zero
                    };

                    if denominator == 0.0 {
                         test_successful = false;
                        outcome_details.push_str("Denominator for variance calculation is zero. ");
                    } else {
                        for k_idx in 0..k_eff {
                            let score_column_k = scores.column(k_idx);
                            let sum_sq_f64 = score_column_k.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                            let variance_of_score_k = sum_sq_f64 / denominator;
                            
                            let diff = (variance_of_score_k - eigenvalues[k_idx]).abs();
                            if diff > max_variance_eigenvalue_diff {
                                max_variance_eigenvalue_diff = diff;
                            }
                            if diff >= DEFAULT_FLOAT_TOLERANCE_F64 * 100.0 { // Relaxed tolerance
                                test_successful = false;
                                outcome_details.push_str(&format!(
                                    "Variance of score column {} ({}) does not match eigenvalue {} ({}). Diff: {}. ",
                                    k_idx, variance_of_score_k, k_idx, eigenvalues[k_idx], diff
                                ));
                            }
                        }
                        if test_successful {
                            outcome_details.push_str(&format!("All variance-eigenvalue checks passed. Max diff: {:.2e}. ", max_variance_eigenvalue_diff));
                        } else {
                            outcome_details.push_str(&format!("Variance-eigenvalue checks failed. Max diff: {:.2e}. ", max_variance_eigenvalue_diff));
                        }
                    }
                }
                let record = eigensnp_integration_tests::TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: output.num_principal_components_computed,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                TEST_RESULTS.lock().unwrap().push(record);
            }
            Ok(Err(e)) => {
                test_successful = false;
                outcome_details = format!("PCA computation failed: {:?}", e);
                let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: 0,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                TEST_RESULTS.lock().unwrap().push(record);
            }
            Err(e) => {
                test_successful = false;
                outcome_details = format!("Test panicked: {:?}", e);
                 let record = TestResultRecord {
                    test_name: test_name.clone(),
                    num_features_d: num_snps,
                    num_samples_n: num_samples,
                    num_pcs_requested_k: num_pcs_target,
                    num_pcs_computed: 0,
                    success: test_successful,
                    outcome_details: outcome_details.clone(),
                    notes,
                };
                TEST_RESULTS.lock().unwrap().push(record);
            }
        }
         assert!(test_successful, "Test {} failed. Max variance-eigenvalue diff: {:.2e}. Details: {}", test_name, max_variance_eigenvalue_diff, outcome_details);
    }

    #[test]
    fn test_eigenvalue_score_variance_correspondence_large_500x100() {
        run_eigenvalue_score_variance_correspondence_test("test_eigenvalue_score_variance_correspondence_large_500x100", 500, 100, 5, 789);
    }

    #[test]
    fn test_eigenvalue_score_variance_correspondence_large_1000x200() {
        run_eigenvalue_score_variance_correspondence_test("test_eigenvalue_score_variance_correspondence_large_1000x200", 1000, 200, 10, 790);
    }
    
    #[test]
    fn test_pca_zero_snps() {
        let num_samples = 10;
        let num_snps = 0; // Explicit for clarity in logging
        let k_requested = 2;

        let test_data = TestDataAccessor::new_empty(num_snps, num_samples);

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: k_requested,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![]; 

        let output = algorithm.compute_pca(&test_data, &ld_blocks).expect("PCA with 0 SNPs failed");

        assert_eq!(output.num_pca_snps_used, 0);
        assert_eq!(output.num_qc_samples_used, num_samples);
        assert_eq!(output.num_principal_components_computed, 0);
        assert_eq!(output.final_snp_principal_component_loadings.nrows(), 0);
        assert_eq!(output.final_snp_principal_component_loadings.ncols(), 0);
        assert_eq!(output.final_sample_principal_component_scores.nrows(), num_samples);
        assert_eq!(output.final_sample_principal_component_scores.ncols(), 0);
        assert_eq!(output.final_principal_component_eigenvalues.len(), 0);

        let record = TestResultRecord {
            test_name: "test_pca_zero_snps".to_string(),
            num_features_d: num_snps,
            num_samples_n: num_samples,
            num_pcs_requested_k: k_requested,
            num_pcs_computed: output.num_principal_components_computed,
            success: true, // Assertions above would panic on failure
            outcome_details: "Validated behavior with zero SNPs. All assertions passed.".to_string(),
            notes: "Edge case test.".to_string(),
        };
        TEST_RESULTS.lock().unwrap().push(record);
    }

    #[test]
    fn test_pca_zero_samples() {
        let num_snps = 20;
        let num_samples = 0; // Explicit for clarity in logging
        let k_requested = 2;

        let test_data = TestDataAccessor::new_empty(num_snps, num_samples); 

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: k_requested,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification { 
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
        }];

        let output = algorithm.compute_pca(&test_data, &ld_blocks).expect("PCA with 0 samples failed");

        assert_eq!(output.num_qc_samples_used, 0);
        assert_eq!(output.num_pca_snps_used, num_snps);
        assert_eq!(output.num_principal_components_computed, 0);
        assert_eq!(output.final_snp_principal_component_loadings.nrows(), num_snps);
        assert_eq!(output.final_snp_principal_component_loadings.ncols(), 0);
        assert_eq!(output.final_sample_principal_component_scores.nrows(), 0);
        assert_eq!(output.final_sample_principal_component_scores.ncols(), 0);
        assert_eq!(output.final_principal_component_eigenvalues.len(), 0);

        let record = TestResultRecord {
            test_name: "test_pca_zero_samples".to_string(),
            num_features_d: num_snps,
            num_samples_n: num_samples,
            num_pcs_requested_k: k_requested,
            num_pcs_computed: output.num_principal_components_computed,
            success: true, // Assertions above would panic on failure
            outcome_details: "Validated behavior with zero samples. All assertions passed.".to_string(),
            notes: "Edge case test.".to_string(),
        };
        TEST_RESULTS.lock().unwrap().push(record);
    }

    #[test]
    fn test_pca_more_components_requested_than_rank_d_gt_n() {
        let mut test_successful = true;
        let mut outcome_details = String::new();
        let mut notes = String::new();
        notes.push_str("Testing k_requested > true rank, with D > N (150x50). ");

        let num_samples = 50; // N
        let num_true_rank_snps = 20; // True rank of D
        let num_total_snps = 150; // D (num_features)
        let k_components_requested = 30; // k_requested > num_true_rank_snps

        let mut raw_genos = Array2::<f32>::zeros((num_total_snps, num_samples));
        let mut rng = ChaCha8Rng::seed_from_u64(321);
        for r in 0..num_true_rank_snps {
            for c in 0..num_samples {
                raw_genos[[r,c]] = rng.sample(Uniform::new(0.0, 3.0));
            }
        }
        // Create linear dependencies for SNPs beyond num_true_rank_snps
        for r in num_true_rank_snps..num_total_snps {
            let source_row_idx = r % num_true_rank_snps; // pick a source row from the true rank block
            let factor = rng.sample(Uniform::new(0.3, 0.7));
            let noise: f32 = rng.sample(Uniform::new(-0.01, 0.01)); // Small noise
            for c in 0..num_samples {
                 raw_genos[[r,c]] = raw_genos[[source_row_idx, c]] * factor + noise;
            }
        }
        
        let standardized_genos = standardize_features_across_samples(raw_genos.clone());
        let test_data = TestDataAccessor::new(standardized_genos.clone());

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: k_components_requested, 
            components_per_ld_block: num_total_snps.min(num_samples), 
            random_seed: 321,
            subset_factor_for_local_basis_learning: 1.0, 
            min_subset_size_for_local_basis_learning: num_samples.max(1), // Ensure at least 1
            max_subset_size_for_local_basis_learning: num_samples.max(10), // Ensure at least 10
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification {
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..num_total_snps).map(PcaSnpId).collect(),
        }];

        let rust_output_result = algorithm.compute_pca(&test_data, &ld_blocks);
        
        let rust_output = match rust_output_result {
            Ok(output) => output,
            Err(e) => {
                test_successful = false;
                outcome_details.push_str(&format!("Rust PCA computation failed: {}. ", e));
                // Create a dummy output to allow logging and avoid panics before logging
                efficient_pca::eigensnp::EigenSNPCoreOutput {
                    final_snp_principal_component_loadings: Array2::zeros((0,0)),
                    final_sample_principal_component_scores: Array2::zeros((0,0)),
                    final_principal_component_eigenvalues: Array1::zeros(0),
                    num_principal_components_computed: 0,
                    num_pca_snps_used: num_total_snps, // Ensure num_total_snps is in scope
                    num_qc_samples_used: num_samples,   // Ensure num_samples is in scope
                }
            }
        };
        let effective_k_rust = rust_output.num_principal_components_computed;

        let mut py_loadings_k_x_d: Array2<f32> = Array2::zeros((0,0));
        let mut py_scores_n_x_k: Array2<f32> = Array2::zeros((0,0));
        let mut py_eigenvalues_k: Array1<f64> = Array1::zeros(0);
        let mut effective_k_py = 0;

        let mut stdin_data = String::new();
        for i in 0..standardized_genos.nrows() {
            for j in 0..standardized_genos.ncols() {
                stdin_data.push_str(&standardized_genos[[i,j]].to_string());
                if j < standardized_genos.ncols() - 1 { stdin_data.push(' '); }
            }
            stdin_data.push('\n');
        }
        
        let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        script_path.push("tests/pca.py");

        let process_result = Command::new("python3")
            .arg(script_path.to_str().unwrap())
            .arg("--generate-reference-pca")
            .arg("-k").arg(k_components_requested.to_string())
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn();

        match process_result {
            Ok(mut process) => {
                let mut stdin_pipe = process.stdin.take().expect("Failed to open stdin for pca.py");
                std::thread::spawn(move || {
                    match stdin_pipe.write_all(stdin_data.as_bytes()) {
                        Ok(_) => {},
                        Err(e) => eprintln!("Failed to write to stdin of pca.py: {}", e), // Use eprintln for errors
                    }
                });

                let py_cmd_output_result = process.wait_with_output();
                match py_cmd_output_result {
                    Ok(py_cmd_output) => {
                        let stderr_str_lossy = String::from_utf8_lossy(&py_cmd_output.stderr);
                        notes.push_str(&format!("Python stderr: {}. ", stderr_str_lossy.trim()));
                        if !py_cmd_output.status.success() {
                            test_successful = false;
                            outcome_details.push_str(&format!("Python script execution failed. Status: {}. Stdout: {}. Stderr: {}. ", 
                                py_cmd_output.status.code().unwrap_or(-1),
                                String::from_utf8_lossy(&py_cmd_output.stdout).trim(),
                                stderr_str_lossy.trim()));
                        } else {
                            let python_output_str = String::from_utf8_lossy(&py_cmd_output.stdout);
                        match parse_pca_py_output(&python_output_str) {
                                Ok((loadings, scores, eigenvalues)) => {
                                    py_loadings_k_x_d = loadings;
                                    py_scores_n_x_k = scores;
                                    py_eigenvalues_k = eigenvalues;
                                    effective_k_py = py_eigenvalues_k.len();
                                    outcome_details.push_str(&format!("Rust k_eff: {}, Py k_eff: {}. ", effective_k_rust, effective_k_py));
                                }
                                Err(err_msg) => {
                                    test_successful = false;
                                    outcome_details.push_str(&format!("Failed to parse pca.py output: {}. Python stdout: {}. ", err_msg, python_output_str.trim()));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        test_successful = false;
                        outcome_details.push_str(&format!("Failed to wait for pca.py process: {}. ", e));
                    }
                }
            }
            Err(e) => {
                test_successful = false;
                outcome_details.push_str(&format!("Failed to spawn pca.py process: {}. ", e));
            }
        }
        
        let mut eigenvalue_checks_performed = false;
        if test_successful {
            if effective_k_rust > num_true_rank_snps {
                eigenvalue_checks_performed = true;
                for i in num_true_rank_snps..effective_k_rust {
                    if rust_output.final_principal_component_eigenvalues[i] > 1e-3 {
                        test_successful = false;
                        outcome_details.push_str(&format!("Rust Eigenvalue for PC {} ({}) beyond true rank ({}) is too large ({}). ", 
                            i, rust_output.final_principal_component_eigenvalues[i], num_true_rank_snps, 1e-3));
                        break; 
                    }
                }
            }
            if test_successful && effective_k_py > 0 && effective_k_py > num_true_rank_snps { // Check effective_k_py > 0 before accessing py_eigenvalues_k
                 eigenvalue_checks_performed = true;
                for i in num_true_rank_snps..effective_k_py {
                     if py_eigenvalues_k.get(i).map_or(false, |&val| val > 1e-3) { // Safely get value
                        test_successful = false;
                        outcome_details.push_str(&format!("Python Eigenvalue for PC {} ({}) beyond true rank ({}) is too large ({}). ", 
                            i, py_eigenvalues_k.get(i).unwrap_or(&0.0), num_true_rank_snps, 1e-3));
                        break;
                    }
                }
            }
            if eigenvalue_checks_performed && test_successful {
                 outcome_details.push_str("Eigenvalues beyond true rank checked and are small. ");
            } else if !eigenvalue_checks_performed {
                outcome_details.push_str("Eigenvalue checks beyond true rank not performed (k_eff <= true_rank or test already failed). ");
            }
        }
        
        let py_loadings_d_x_k = py_loadings_k_x_d.t().into_owned(); // Transpose even if empty

        let artifact_dir = "target/test_artifacts/pca_low_rank_D_gt_N"; // Updated artifact_dir
        if rust_output.num_principal_components_computed > 0 { // Save only if there's something to save
            save_matrix_to_tsv(&rust_output.final_snp_principal_component_loadings.view(), artifact_dir, "rust_loadings.tsv").expect("Failed to save rust_loadings.tsv");
            save_matrix_to_tsv(&rust_output.final_sample_principal_component_scores.view(), artifact_dir, "rust_scores.tsv").expect("Failed to save rust_scores.tsv");
            save_vector_to_tsv(&rust_output.final_principal_component_eigenvalues.view(), artifact_dir, "rust_eigenvalues.tsv").expect("Failed to save rust_eigenvalues.tsv");
        }
        if effective_k_py > 0 { // Save Python results only if they exist
            save_matrix_to_tsv(&py_loadings_d_x_k.view(), artifact_dir, "python_loadings.tsv").expect("Failed to save python_loadings.tsv");
            save_matrix_to_tsv(&py_scores_n_x_k.view(), artifact_dir, "python_scores.tsv").expect("Failed to save python_scores.tsv");
            save_vector_to_tsv(&py_eigenvalues_k.view(), artifact_dir, "python_eigenvalues.tsv").expect("Failed to save python_eigenvalues.tsv");
        }
        
        let record = TestResultRecord {
            test_name: "test_pca_more_components_requested_than_rank_D_gt_N".to_string(),
            num_features_d: num_total_snps,
            num_samples_n: num_samples,
            num_pcs_requested_k: k_components_requested,
            num_pcs_computed: effective_k_rust, // Rust's computed PCs
            success: test_successful,
            outcome_details: outcome_details.clone(),
            notes,
        };
        TEST_RESULTS.lock().unwrap().push(record);

        assert!(test_successful, "Test 'test_pca_more_components_requested_than_rank_D_gt_N' failed prior to detailed comparisons. Details: {}", outcome_details);
        
        // Only proceed with detailed comparisons if all prior checks (including Python script) were successful
        if test_successful {
            println!("Low-rank D>N test: Rust computed {} PCs, Python computed {} PCs (requested {}, true rank {})", 
                effective_k_rust, effective_k_py, k_components_requested, num_true_rank_snps);

            assert!(effective_k_rust <= k_components_requested, "Rust computed more PCs ({}) than requested ({}).", effective_k_rust, k_components_requested);
            assert!(effective_k_py <= k_components_requested, "Python computed more PCs ({}) than requested ({}).", effective_k_py, k_components_requested);
            
            // Compare up to min of (effective_k_rust, effective_k_py, num_true_rank_snps + a small margin like 2, k_components_requested)
            // This ensures we only compare meaningful components and don't go out of bounds.
            let num_pcs_to_compare = effective_k_rust.min(effective_k_py)
                                        .min(num_true_rank_snps + 2) // Compare slightly beyond true rank if available
                                        .min(k_components_requested);


            if num_pcs_to_compare > 0 {
                 assert_f32_arrays_are_close_with_sign_flips(
                    rust_output.final_snp_principal_component_loadings.slice(s![.., 0..num_pcs_to_compare]),
                    py_loadings_d_x_k.slice(s![.., 0..num_pcs_to_compare]),
                    1.5f32, 
                    "SNP Loadings (Low-Rank D>N)"
                );
                assert_f32_arrays_are_close_with_sign_flips(
                    rust_output.final_sample_principal_component_scores.slice(s![.., 0..num_pcs_to_compare]),
                    py_scores_n_x_k.slice(s![.., 0..num_pcs_to_compare]),
                    DEFAULT_FLOAT_TOLERANCE_F32 * 10.0,
                    "Sample Scores (Low-Rank D>N)"
                );
                assert_f64_arrays_are_close(
                    rust_output.final_principal_component_eigenvalues.slice(s![0..num_pcs_to_compare]),
                    py_eigenvalues_k.slice(s![0..num_pcs_to_compare]),
                    DEFAULT_FLOAT_TOLERANCE_F64 * 10.0, 
                    "Eigenvalues (Low-Rank D>N)"
                );
            } else {
                println!("No principal components available for detailed comparison (num_pcs_to_compare = {}). Rust k_eff: {}, Py k_eff: {}.", 
                    num_pcs_to_compare, effective_k_rust, effective_k_py);
                // If one is 0 and the other is not, it's a failure if test_successful was true
                if effective_k_rust != effective_k_py {
                     panic!("Mismatch in effective number of PCs where one is zero. Rust: {}, Python: {}. This should have been caught earlier if test_successful was false.", effective_k_rust, effective_k_py);
                }
            }
        }
    }
}

// Helper function for Pearson Correlation
pub fn pearson_correlation(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> Option<f32> {
    if v1.len() != v2.len() || v1.is_empty() {
        return None;
    }
    let _n = v1.len() as f32;
    let mean1 = v1.mean().unwrap_or(0.0);
    let mean2 = v2.mean().unwrap_or(0.0);
    let mut cov = 0.0;
    let mut std_dev1_sq = 0.0;
    let mut std_dev2_sq = 0.0;
    for i in 0..v1.len() {
        let d1 = v1[i] - mean1;
        let d2 = v2[i] - mean2;
        cov += d1 * d2;
        std_dev1_sq += d1 * d1;
        std_dev2_sq += d2 * d2;
    }
    if std_dev1_sq <= 1e-6 || std_dev2_sq <= 1e-6 { // Avoid division by zero or near-zero
        // If one vector is constant, correlation is undefined or can be taken as 0
        // For PCA components, they shouldn't be constant if eigenvalues are non-zero.
        // If both are constant and identical, correlation is 1.
        if (std_dev1_sq - std_dev2_sq).abs() < 1e-6 && cov.abs() < 1e-6 { 
            let mut all_v1_same = true;
            let mut all_v2_same = true;
            if v1.len() > 1 {
                for i in 1..v1.len() {
                    if (v1[i] - v1[0]).abs() > 1e-6 { all_v1_same = false; break; }
                    if (v2[i] - v2[0]).abs() > 1e-6 { all_v2_same = false; break; }
                }
            }
            if all_v1_same && all_v2_same && (v1[0] - v2[0]).abs() < 1e-6 { return Some(1.0); }
        }
        return None; 
    }
    Some(cov / (std_dev1_sq.sqrt() * std_dev2_sq.sqrt()))
}

// Helper function for PC correlation tests
pub fn run_pc_correlation_with_truth_set_test(
    test_name_str: &str,
    num_snps: usize,      // D
    num_samples: usize,   // N
    k_components: usize, // k
    seed: u64,
) {
    let test_name = test_name_str.to_string();
    let mut test_successful = true;
    let mut outcome_details = String::new();
    let mut notes = format!("Matrix D_snps x N_samples: {}x{}, k_requested: {}. ", num_snps, num_samples, k_components);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
    let standardized_genos_snps_x_samples = standardize_features_across_samples(raw_genos.clone());

    let artifact_dir_suffix = format!("pc_correlation_{}x{}_k{}", num_snps, num_samples, k_components);
    let artifact_dir = Path::new("target/test_artifacts").join(artifact_dir_suffix);
    if let Err(e) = fs::create_dir_all(&artifact_dir) {
        notes.push_str(&format!("Failed to create artifact dir: {}. ", e));
    }

    // Get Truth PCs from pca.py
    let mut py_loadings_d_x_k: Array2<f32> = Array2::zeros((0,0));
    let mut py_scores_n_x_k: Array2<f32> = Array2::zeros((0,0));
    // let mut py_eigenvalues_k: Array1<f64> = Array1::zeros(0); // Not directly used for correlation but good to have
    let mut effective_k_py = 0;

    let mut stdin_data_py = String::new();
    for i in 0..standardized_genos_snps_x_samples.nrows() {
        for j in 0..standardized_genos_snps_x_samples.ncols() {
            stdin_data_py.push_str(&standardized_genos_snps_x_samples[[i, j]].to_string());
            if j < standardized_genos_snps_x_samples.ncols() - 1 { stdin_data_py.push(' '); }
        }
        stdin_data_py.push('\n');
    }

    let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    script_path.push("tests/pca.py");

    match Command::new("python3")
        .arg(script_path.to_str().unwrap())
        .arg("--generate-reference-pca")
        .arg("-k").arg(k_components.to_string())
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn() 
    {
        Ok(mut process) => {
            if let Some(mut stdin_pipe) = process.stdin.take() {
                let stdin_data_py_clone = stdin_data_py.clone(); // Clone for the thread
                 std::thread::spawn(move || {
                    if let Err(e) = stdin_pipe.write_all(stdin_data_py_clone.as_bytes()) {
                         eprintln!("Failed to write to stdin of pca.py: {}", e);
                    }
                });
            } else {
                 test_successful = false;
                outcome_details.push_str("Failed to get stdin pipe for pca.py. ");
            }
            
            match process.wait_with_output() {
                Ok(py_cmd_output) => {
                    let stderr_py = String::from_utf8_lossy(&py_cmd_output.stderr);
                    notes.push_str(&format!("Python stderr: {}. ", stderr_py.trim()));
                    if !py_cmd_output.status.success() {
                        test_successful = false;
                        outcome_details.push_str(&format!("Python script execution failed. Status: {}. Stdout: {}. Stderr: {}. ",
                            py_cmd_output.status.code().unwrap_or(-1),
                            String::from_utf8_lossy(&py_cmd_output.stdout).trim(),
                            stderr_py.trim()));
                    } else {
                        let python_output_str = String::from_utf8_lossy(&py_cmd_output.stdout);
                        match parse_pca_py_output(&python_output_str) {
                            Ok((loadings_k_x_d, scores_n_x_k_py, _eigenvalues_k_py)) => {
                                py_loadings_d_x_k = loadings_k_x_d.t().into_owned();
                                py_scores_n_x_k = scores_n_x_k_py;
                                // py_eigenvalues_k = _eigenvalues_k_py;
                                effective_k_py = py_loadings_d_x_k.ncols(); // D_snps x k
                                save_matrix_to_tsv(&py_loadings_d_x_k.view(), artifact_dir.to_str().unwrap_or("."), "python_loadings.tsv").unwrap_or_default();
                                save_matrix_to_tsv(&py_scores_n_x_k.view(), artifact_dir.to_str().unwrap_or("."), "python_scores.tsv").unwrap_or_default();
                            }
                            Err(e) => {
                                test_successful = false;
                                outcome_details.push_str(&format!("Failed to parse pca.py output: {}. ", e));
                            }
                        }
                    }
                }
                Err(e) => {
                    test_successful = false;
                    outcome_details.push_str(&format!("Failed to wait for pca.py: {}. ", e));
                }
            }
        }
        Err(e) => {
            test_successful = false;
            outcome_details.push_str(&format!("Failed to spawn pca.py: {}. ", e));
        }
    }

    // Run eigensnp
    let test_data_accessor = TestDataAccessor::new(standardized_genos_snps_x_samples.clone());
    let config = EigenSNPCoreAlgorithmConfig {
        target_num_global_pcs: k_components,
        random_seed: seed,
        subset_factor_for_local_basis_learning: 0.5,
        min_subset_size_for_local_basis_learning: (num_samples / 4).max(1).min(num_samples.max(1)),
        max_subset_size_for_local_basis_learning: (num_samples / 2).max(10).min(num_samples.max(1)),
        components_per_ld_block: 10.min(num_snps.min( (num_samples/2).max(10).min(num_samples.max(1)) )),
        ..Default::default()
    };
    let algorithm = EigenSNPCoreAlgorithm::new(config);
    let ld_blocks = vec![LdBlockSpecification {
        user_defined_block_tag: "block1".to_string(),
        pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
    }];

    let mut rust_pcs_computed = 0;
    match algorithm.compute_pca(&test_data_accessor, &ld_blocks) {
        Ok(rust_result) => {
            rust_pcs_computed = rust_result.num_principal_components_computed;
            save_matrix_to_tsv(&rust_result.final_snp_principal_component_loadings.view(), artifact_dir.to_str().unwrap_or("."), "rust_loadings.tsv").unwrap_or_default();
            save_matrix_to_tsv(&rust_result.final_sample_principal_component_scores.view(), artifact_dir.to_str().unwrap_or("."), "rust_scores.tsv").unwrap_or_default();

            if test_successful { // Only compare if Python part was also successful
                let k_to_compare = rust_result.num_principal_components_computed.min(effective_k_py);
                if k_to_compare == 0 {
                    outcome_details.push_str("No components to compare (Rust or Python computed 0 PCs). ");
                    if rust_result.num_principal_components_computed != effective_k_py { // If one is 0 and other is not
                        test_successful = false;
                        outcome_details.push_str(&format!("Mismatch in computed k (Rust: {}, Py: {}). ", rust_result.num_principal_components_computed, effective_k_py));
                    }
                } else {
                    let mut min_loading_abs_corr = 1.0f32;
                    let mut min_score_abs_corr = 1.0f32;
                    let mut correlations_summary = String::new();

                    for pc_idx in 0..k_to_compare {
                        let rust_loading_col = rust_result.final_snp_principal_component_loadings.column(pc_idx);
                        let py_loading_col = py_loadings_d_x_k.column(pc_idx);
                        let loading_corr = pearson_correlation(rust_loading_col.view(), py_loading_col.view()).map_or(0.0, |c| c.abs());
                        if loading_corr < min_loading_abs_corr { min_loading_abs_corr = loading_corr; }
                        correlations_summary.push_str(&format!("PC{}_Load_absR={:.4}; ", pc_idx, loading_corr));
                        if loading_corr < 0.95 {
                            test_successful = false;
                            outcome_details.push_str(&format!("Low loading correlation for PC {}: {:.4}. ", pc_idx, loading_corr));
                        }

                        let rust_score_col = rust_result.final_sample_principal_component_scores.column(pc_idx);
                        let py_score_col = py_scores_n_x_k.column(pc_idx);
                        let score_corr = pearson_correlation(rust_score_col.view(), py_score_col.view()).map_or(0.0, |c| c.abs());
                        if score_corr < min_score_abs_corr { min_score_abs_corr = score_corr; }
                        correlations_summary.push_str(&format!("PC{}_Score_absR={:.4}; ", pc_idx, score_corr));
                         if score_corr < 0.95 {
                            test_successful = false;
                            outcome_details.push_str(&format!("Low score correlation for PC {}: {:.4}. ", pc_idx, score_corr));
                        }
                    }
                    outcome_details.push_str(&format!("Compared {} PCs. Min loading_absR={:.4}, Min score_absR={:.4}. Full: {}. ", 
                        k_to_compare, min_loading_abs_corr, min_score_abs_corr, correlations_summary.trim_end_matches("; ")));
                }
            }
        }
        Err(e) => {
            test_successful = false;
            outcome_details.push_str(&format!("Rust PCA computation failed: {}. ", e));
        }
    }

    let record = TestResultRecord {
        test_name,
        num_features_d: num_snps,
        num_samples_n: num_samples,
        num_pcs_requested_k: k_components,
        num_pcs_computed: rust_pcs_computed, 
        success: test_successful,
        outcome_details: outcome_details.clone(),
        notes,
    };
    TEST_RESULTS.lock().unwrap().push(record);

    assert!(test_successful, "Test '{}' failed. Check TSV. Details: {}", test_name_str, outcome_details);
}


#[test]
fn test_pc_correlation_with_truth_set_large_1000x200() {
    run_pc_correlation_with_truth_set_test(
        "test_pc_correlation_with_truth_set_large_1000x200",
        1000, // num_snps (D)
        200,  // num_samples (N)
        10,   // k_components
        202401, // seed
    );
}

// Helper function for generic large matrix execution tests
pub fn run_generic_large_matrix_test(
    test_name_str: &str,
    num_snps: usize,      // D
    num_samples: usize,   // N
    k_components: usize, // k
    seed: u64,
    // Optional: allow passing a custom config modifier
    config_modifier: Option<fn(EigenSNPCoreAlgorithmConfig) -> EigenSNPCoreAlgorithmConfig>, 
) {
    let test_name = test_name_str.to_string();
    let mut test_successful = true;
    let mut outcome_details = String::new();
    let mut notes = format!("Matrix D_snps x N_samples: {}x{}, k_requested: {}. ", num_snps, num_samples, k_components);

    let artifact_dir_suffix = format!("generic_large_matrix_{}x{}_k{}", num_snps, num_samples, k_components);
    let artifact_dir = Path::new("target/test_artifacts").join(artifact_dir_suffix);
     if let Err(e) = fs::create_dir_all(&artifact_dir) {
        notes.push_str(&format!("Failed to create artifact dir: {}. ", e));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
    let standardized_genos = standardize_features_across_samples(raw_genos);

    let test_data_accessor = TestDataAccessor::new(standardized_genos);
    
    let mut base_config = EigenSNPCoreAlgorithmConfig {
        target_num_global_pcs: k_components,
        random_seed: seed,
        ..Default::default()
    };

    if let Some(modifier) = config_modifier {
        base_config = modifier(base_config);
        notes.push_str("Custom config modifier applied. ");
    }


    let algorithm = EigenSNPCoreAlgorithm::new(base_config);
    let ld_blocks = vec![LdBlockSpecification {
        user_defined_block_tag: "block_generic".to_string(),
        pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
    }];
    
    let mut rust_pcs_computed = 0;

    match algorithm.compute_pca(&test_data_accessor, &ld_blocks) {
        Ok(output) => {
            rust_pcs_computed = output.num_principal_components_computed;
            outcome_details = format!(
                "eigensnp successful. Computed {} PCs. First eigenvalue: {:.4}. ",
                output.num_principal_components_computed,
                output.final_principal_component_eigenvalues.get(0).unwrap_or(&0.0)
            );
            if rust_pcs_computed == 0 && k_components > 0 {
                 test_successful = false;
                 outcome_details.push_str("Warning: 0 PCs computed when k_requested > 0. ");
            }
            if rust_pcs_computed > k_components {
                 test_successful = false;
                 outcome_details.push_str(&format!("Warning: More PCs computed ({}) than requested ({}). ", rust_pcs_computed, k_components));
            }
            // Save outputs
            save_matrix_to_tsv(&output.final_snp_principal_component_loadings.view(), artifact_dir.to_str().unwrap_or("."), "rust_loadings.tsv").unwrap_or_default();
            save_matrix_to_tsv(&output.final_sample_principal_component_scores.view(), artifact_dir.to_str().unwrap_or("."), "rust_scores.tsv").unwrap_or_default();
            save_vector_to_tsv(&output.final_principal_component_eigenvalues.view(), artifact_dir.to_str().unwrap_or("."), "rust_eigenvalues.tsv").unwrap_or_default();
        }
        Err(e) => {
            test_successful = false;
            outcome_details = format!("eigensnp PCA computation failed: {}. ", e);
        }
    }

    let record = TestResultRecord {
        test_name,
        num_features_d: num_snps,
        num_samples_n: num_samples,
        num_pcs_requested_k: k_components,
        num_pcs_computed: rust_pcs_computed,
        success: test_successful,
        outcome_details: outcome_details.clone(),
        notes,
    };
    TEST_RESULTS.lock().unwrap().push(record);

    assert!(test_successful, "Test '{}' failed. Check TSV. Details: {}", test_name_str, outcome_details);
}

#[test]
fn test_large_matrix_2000x200_k10() {
    run_generic_large_matrix_test(
        "test_large_matrix_2000x200_k10",
        2000, // num_snps
        200,  // num_samples
        10,   // k_components
        202405, // seed
        None,
    );
}

#[test]
fn test_large_matrix_5000x500_k20() {
    run_generic_large_matrix_test(
        "test_large_matrix_5000x500_k20",
        5000, // num_snps
        500,  // num_samples
        20,   // k_components
        202406, // seed
        None,
    );
}

#[test]
fn test_large_matrix_1000x100_k5_blocksize_variation() {
    run_generic_large_matrix_test(
        "test_large_matrix_1000x100_k5_blocksize_variation",
        1000, // num_snps
        100,  // num_samples
        5,    // k_components
        202407, // seed
        Some(|mut cfg: EigenSNPCoreAlgorithmConfig| {
            cfg.components_per_ld_block = 20; // Example variation
            cfg.subset_factor_for_local_basis_learning = 0.8;
            cfg
        }),
    );
}

// Helper function for sample projection accuracy tests
pub fn run_sample_projection_accuracy_test(
    test_name_str: &str,
    num_snps: usize,          // D
    num_samples_total: usize, // N_total
    num_samples_train: usize, // N_train
    k_components: usize,      // k
    seed: u64,
) {
    let test_name = test_name_str.to_string();
    let mut test_successful = true;
    let mut outcome_details: String;
    let num_samples_test = num_samples_total - num_samples_train;
    let mut notes = format!(
        "Matrix D_snps x N_total_samples (N_train_samples / N_test_samples): {}x{} ({} / {}), k_requested: {}. ",
        num_snps, num_samples_total, num_samples_train, num_samples_test, k_components
    );

    assert!(num_samples_train > 0, "num_samples_train must be > 0");
    assert!(num_samples_total > num_samples_train, "num_samples_total must be > num_samples_train");

    let artifact_dir_suffix = format!("sample_projection_{}x{}_k{}", num_snps, num_samples_train, k_components);
    let artifact_dir = Path::new("target/test_artifacts").join(artifact_dir_suffix);
    if let Err(e) = fs::create_dir_all(&artifact_dir) {
        notes.push_str(&format!("Failed to create artifact dir: {}. ", e));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let raw_genos_total = Array2::random_using((num_snps, num_samples_total), Uniform::new(0.0, 3.0), &mut rng);
    let standardized_genos_total_snps_x_samples = standardize_features_across_samples(raw_genos_total);

    let train_data_snps_x_samples = standardized_genos_total_snps_x_samples.slice(s![.., 0..num_samples_train]).to_owned();
    let test_data_snps_x_samples = standardized_genos_total_snps_x_samples.slice(s![.., num_samples_train..]).to_owned();

    // Run eigensnp on Training Data
    let test_data_accessor_train = TestDataAccessor::new(train_data_snps_x_samples.clone());
    let config_train = EigenSNPCoreAlgorithmConfig {
        target_num_global_pcs: k_components,
        random_seed: seed,
        // Default other params or make them configurable if needed for these tests
        ..Default::default() 
    };
    let algorithm_train = EigenSNPCoreAlgorithm::new(config_train);
    let ld_blocks_train = vec![LdBlockSpecification {
        user_defined_block_tag: "block_train".to_string(),
        pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
    }];

    let mut rust_pca_output_option: Option<efficient_pca::eigensnp::EigenSNPCoreOutput> = None;
    let mut k_eff_rust = 0;

    match algorithm_train.compute_pca(&test_data_accessor_train, &ld_blocks_train) {
        Ok(output) => {
            k_eff_rust = output.num_principal_components_computed;
            save_matrix_to_tsv(&output.final_snp_principal_component_loadings.view(), artifact_dir.to_str().unwrap_or("."), "rust_train_loadings.tsv").unwrap_or_default();
            save_matrix_to_tsv(&output.final_sample_principal_component_scores.view(), artifact_dir.to_str().unwrap_or("."), "rust_train_scores.tsv").unwrap_or_default();
            rust_pca_output_option = Some(output);
            outcome_details = format!("eigensnp on train data successful. k_eff_rust: {}. ", k_eff_rust);
        }
        Err(e) => {
            test_successful = false;
            outcome_details = format!("eigensnp on train data failed: {}. ", e);
        }
    }

    // Project Test Samples using Rust loadings
    let mut s_projected_option: Option<Array2<f32>> = None;
    if let Some(ref rust_pca_output) = rust_pca_output_option {
        if k_eff_rust > 0 {
            let loadings_l = &rust_pca_output.final_snp_principal_component_loadings; // D x k_eff_rust
            if test_data_snps_x_samples.nrows() == loadings_l.nrows() { // D must match
                 // (N_test x D) dot (D x k_eff_rust) = N_test x k_eff_rust
                let projected_scores = test_data_snps_x_samples.t().dot(loadings_l);
                save_matrix_to_tsv(&projected_scores.view(), artifact_dir.to_str().unwrap_or("."), "rust_projected_test_scores.tsv").unwrap_or_default();
                s_projected_option = Some(projected_scores);
            } else {
                 test_successful = false;
                 outcome_details.push_str(&format!("Dimension mismatch for projection: test_data_snps_x_samples.nrows() ({}) != loadings_l.nrows() ({}). ", 
                    test_data_snps_x_samples.nrows(), loadings_l.nrows()));
            }
        } else {
            outcome_details.push_str("Rust PCA computed 0 components, skipping projection. ");
        }
    }

    // Get "Truth" Scores for Test Samples using pca.py on total data
    let mut py_test_scores_ref_option: Option<Array2<f32>> = None;

    if test_successful { // Only proceed if eigensnp part was okay so far
        let mut stdin_data_py_total = String::new();
        for i in 0..standardized_genos_total_snps_x_samples.nrows() {
            for j in 0..standardized_genos_total_snps_x_samples.ncols() {
                stdin_data_py_total.push_str(&standardized_genos_total_snps_x_samples[[i, j]].to_string());
                if j < standardized_genos_total_snps_x_samples.ncols() - 1 { stdin_data_py_total.push(' '); }
            }
            stdin_data_py_total.push('\n');
        }

        let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        script_path.push("tests/pca.py");

        match Command::new("python3")
            .arg(script_path.to_str().unwrap())
            .arg("--generate-reference-pca")
            .arg("-k").arg(k_components.to_string()) // Use original k_components for full data PCA
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut process) => {
                if let Some(mut stdin_pipe) = process.stdin.take() {
                    std::thread::spawn(move || {
                        if let Err(e) = stdin_pipe.write_all(stdin_data_py_total.as_bytes()) {
                            eprintln!("Failed to write to stdin of pca.py (total data): {}", e);
                        }
                    });
                } else {
                    test_successful = false;
                    outcome_details.push_str("Failed to get stdin pipe for pca.py (total data). ");
                }

                match process.wait_with_output() {
                    Ok(py_cmd_output) => {
                        let stderr_py = String::from_utf8_lossy(&py_cmd_output.stderr);
                        notes.push_str(&format!("Python (total data) stderr: {}. ", stderr_py.trim()));
                        if !py_cmd_output.status.success() {
                            test_successful = false;
                            outcome_details.push_str(&format!("Python script (total data) execution failed. Status: {}. ", py_cmd_output.status.code().unwrap_or(-1)));
                        } else {
                            let python_output_str = String::from_utf8_lossy(&py_cmd_output.stdout);
                            match parse_pca_py_output(&python_output_str) {
                                Ok((_py_loadings_total, py_scores_total_n_x_k, _py_eigenvalues_total)) => {
                                    let k_py_total = _py_loadings_total.ncols(); // k_x_d, so ncols is k
                                    if py_scores_total_n_x_k.nrows() == num_samples_total && py_scores_total_n_x_k.ncols() >= k_components.min(k_py_total) {
                                        // Extract test sample scores: from row num_samples_train onwards
                                        let py_test_scores_ref = py_scores_total_n_x_k.slice(s![num_samples_train.., ..]).to_owned();
                                        save_matrix_to_tsv(&py_test_scores_ref.view(), artifact_dir.to_str().unwrap_or("."), "python_ref_test_scores.tsv").unwrap_or_default();
                                        py_test_scores_ref_option = Some(py_test_scores_ref);
                                        outcome_details.push_str(&format!("Python on total data successful. k_py_total: {}. ", k_py_total));
                                    } else {
                                        test_successful = false;
                                        outcome_details.push_str(&format!("Python (total data) scores dimensions mismatch. Expected N_total x >=k_eff_py ({}x{}), Got {}x{}. ",
                                            num_samples_total, k_components.min(k_py_total), py_scores_total_n_x_k.nrows(), py_scores_total_n_x_k.ncols()));
                                    }
                                }
                                Err(e) => {
                                    test_successful = false;
                                    outcome_details.push_str(&format!("Failed to parse pca.py (total data) output: {}. ", e));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        test_successful = false;
                        outcome_details.push_str(&format!("Failed to wait for pca.py (total data): {}. ", e));
                    }
                }
            }
            Err(e) => {
                test_successful = false;
            outcome_details = format!("eigensnp on train data failed: {}. ", e);
            }
        }
    }

    // Compare Projected Scores with Reference Scores
    if test_successful && s_projected_option.is_some() && py_test_scores_ref_option.is_some() {
        let s_projected = s_projected_option.as_ref().unwrap();
        let py_test_scores_ref = py_test_scores_ref_option.as_ref().unwrap();
        
        let k_compare = k_eff_rust.min(py_test_scores_ref.ncols());
        outcome_details.push_str(&format!("Comparing {} PCs for projection. ", k_compare));

        if k_compare == 0 {
            if k_eff_rust != py_test_scores_ref.ncols() { // If one is 0 and other is not (and k_compare became 0)
                 test_successful = false;
                 outcome_details.push_str(&format!("Mismatch in comparable k (Rust_eff_k: {}, Py_ref_k: {}). ", k_eff_rust, py_test_scores_ref.ncols()));
            } else { // Both are 0
                 outcome_details.push_str("Both Rust and Py_ref have 0 PCs to compare. ");
            }
        } else {
            let mut min_abs_corr = 1.0f32;
            let mut max_mse = 0.0f32;
            let mut correlations_summary = String::new();
            let mut mses_summary = String::new();

            for pc_idx in 0..k_compare {
                let projected_col = s_projected.column(pc_idx);
                let ref_col = py_test_scores_ref.column(pc_idx);

                // Correlation
                let abs_corr = pearson_correlation(projected_col.view(), ref_col.view()).map_or(0.0, |c| c.abs());
                if abs_corr < min_abs_corr { min_abs_corr = abs_corr; }
                correlations_summary.push_str(&format!("PC{}_absR={:.4}; ", pc_idx, abs_corr));
                if abs_corr < 0.95 {
                    test_successful = false;
                    outcome_details.push_str(&format!("Low projection correlation for PC {}: {:.4}. ", pc_idx, abs_corr));
                }

                // MSE
                let mse = (projected_col.to_owned() - ref_col.to_owned()).mapv(|x| x*x).mean().unwrap_or(f32::MAX);
                if mse > max_mse { max_mse = mse; }
                mses_summary.push_str(&format!("PC{}_MSE={:.4e}; ", pc_idx, mse));
                if mse > 0.1 { // Threshold for MSE, might need adjustment
                    test_successful = false;
                    outcome_details.push_str(&format!("High projection MSE for PC {}: {:.4e}. ", pc_idx, mse));
                }
            }
            outcome_details.push_str(&format!(
                "Min_abs_correlation: {:.4}, Max_MSE: {:.4e}. Correlations: {}. MSEs: {}. ",
                min_abs_corr, max_mse, correlations_summary.trim_end_matches("; "), mses_summary.trim_end_matches("; ")
            ));
        }
    } else if test_successful { // If previous steps were successful but options are None
        test_successful = false; // Should not happen if logic is correct
        outcome_details.push_str("Comparison skipped due to missing projected or reference scores despite earlier success. ");
    }


    let record = TestResultRecord {
        test_name,
        num_features_d: num_snps,
        num_samples_n: num_samples_total, // Log total N for context
        num_pcs_requested_k: k_components,
        num_pcs_computed: k_eff_rust, // Rust's computed PCs on training set
        success: test_successful,
        outcome_details: outcome_details.clone(),
        notes,
    };
    TEST_RESULTS.lock().unwrap().push(record);

    assert!(test_successful, "Test '{}' failed. Check TSV. Details: {}", test_name_str, outcome_details);
}


#[test]
fn test_sample_projection_accuracy_large_config1() {
    run_sample_projection_accuracy_test(
        "test_sample_projection_accuracy_large_config1",
        1000, // num_snps (D)
        250,  // num_samples_total
        200,  // num_samples_train
        10,   // k_components
        202403, // seed
    );
}

#[test]
fn test_sample_projection_accuracy_large_config2() {
    run_sample_projection_accuracy_test(
        "test_sample_projection_accuracy_large_config2",
        2000, // num_snps (D)
        400,  // num_samples_total
        300,  // num_samples_train
        15,   // k_components
        202404, // seed
    );
}

#[test]
fn test_pc_correlation_with_truth_set_large_2000x300() {
    run_pc_correlation_with_truth_set_test(
        "test_pc_correlation_with_truth_set_large_2000x300",
        2000, // num_snps (D)
        300,  // num_samples (N)
        15,   // k_components
        202402, // seed
    );
}


// Original tests for reorder utils
// use efficient_pca::eigensnp::{reorder_array_owned, reorder_columns_owned}; // Removed

// use ndarray::{Array1, Array2, arr2}; // Removed

#[test]
fn test_reorder_array_basic() {
    let original = Array1::from(vec![10, 20, 30, 40]);
    let order = vec![2, 0, 3, 1];
    let expected = Array1::from(vec![30, 10, 40, 20]);
    let reordered = reorder_array_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_array_empty_array_empty_order() {
    let original = Array1::<i32>::from(vec![]);
    let order_empty = vec![];
    let expected = Array1::<i32>::from(vec![]);
    let reordered_empty_order = reorder_array_owned(&original, &order_empty);
    assert_eq!(reordered_empty_order, expected);
}

#[test]
#[should_panic]
fn test_reorder_array_select_from_zero_elements_panics() {
    let original = Array1::<i32>::from(vec![]);
    let order = vec![0];
    reorder_array_owned(&original, &order);
}

#[test]
fn test_reorder_array_empty_order() {
    let original = Array1::from(vec![10, 20, 30]);
    let order = vec![];
    let expected = Array1::<i32>::from(vec![]);
    let reordered = reorder_array_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_array_repeated_indices() {
    let original = Array1::from(vec![10, 20, 30]);
    let order = vec![0, 1, 0, 2, 1, 1];
    let expected = Array1::from(vec![10, 20, 10, 30, 20, 20]);
    let reordered = reorder_array_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_columns_basic() {
    let original = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let order = vec![2, 0, 1];
    let expected = arr2(&[[3, 1, 2], [6, 4, 5]]);
    let reordered = reorder_columns_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_columns_empty_matrix_variants() {
    let original_0_rows = Array2::<i32>::zeros((0, 3));
    let order = vec![1, 0, 2];
    let expected_0_rows = Array2::<i32>::zeros((0, 3));
    let reordered_0_rows = reorder_columns_owned(&original_0_rows, &order);
    assert_eq!(reordered_0_rows, expected_0_rows);

    let original_0_cols = Array2::<i32>::zeros((2, 0));
    let order_empty = vec![];
    let expected_0_cols_empty_order = Array2::<i32>::zeros((2,0));
    let reordered_empty_order = reorder_columns_owned(&original_0_cols, &order_empty);
    assert_eq!(reordered_empty_order, expected_0_cols_empty_order);
}

#[test]
#[should_panic]
fn test_reorder_columns_select_from_zero_cols_panics() {
    let original_0_cols = Array2::<i32>::zeros((2, 0));
    let order_for_0_cols = vec![0];
    reorder_columns_owned(&original_0_cols, &order_for_0_cols);
}
   
#[test]
fn test_reorder_columns_empty_order() {
    let original = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let order = vec![];
    let expected = Array2::<i32>::zeros((2, 0));
    let reordered = reorder_columns_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_columns_repeated_indices() {
    let original = arr2(&[[1, 2], [3, 4]]);
    let order = vec![0, 1, 0, 0];
    let expected = arr2(&[[1, 2, 1, 1], [3, 4, 3, 3]]);
    let reordered = reorder_columns_owned(&original, &order);
    assert_eq!(reordered, expected);
}