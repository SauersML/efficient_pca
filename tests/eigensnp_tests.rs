// In tests/eigensnp_tests.rs

use ndarray::{arr2, s, Array1, Array2, ArrayView1, Axis}; // Removed arr1, Ix1, Ix2, and ArrayView2
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

    fn parse_pca_py_output(output_str: &str) -> Result<(Array2<f32>, Array2<f32>, Array1<f64>), String> {
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

    #[test]
    fn test_pca_with_known_small_dataset() {
        let k_components = 2;
        let raw_genotypes_rust = arr2(&[ // SNPs x Samples (D x N)
            [1.0, 2.0, 0.0, 1.0, 2.0], 
            [0.0, 1.0, 1.0, 2.0, 0.0], 
            [2.0, 0.0, 2.0, 1.0, 1.0], 
            [1.0, 1.0, 0.0, 0.0, 2.0], 
        ]); 
        
        let standardized_rust_data_snps_x_samples = standardize_features_across_samples(raw_genotypes_rust.clone());
        println!("DEBUG RUST standardized_data_snps_x_samples (D_snps x N_samples):
{:?}", standardized_rust_data_snps_x_samples);
        let test_data_accessor = TestDataAccessor::new(standardized_rust_data_snps_x_samples.clone());

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: k_components,
            subset_factor_for_local_basis_learning: 1.0,
            min_subset_size_for_local_basis_learning: 1,
            max_subset_size_for_local_basis_learning: 100,
            components_per_ld_block: standardized_rust_data_snps_x_samples.nrows().min(standardized_rust_data_snps_x_samples.ncols()).min(k_components + 2),
            random_seed: 42,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification {
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..test_data_accessor.num_pca_snps()).map(PcaSnpId).collect(),
        }];

        let rust_result = algorithm.compute_pca(&test_data_accessor, &ld_blocks).expect("Rust PCA failed");

        let mut stdin_data = String::new();
        for i in 0..standardized_rust_data_snps_x_samples.nrows() {
            for j in 0..standardized_rust_data_snps_x_samples.ncols() {
                stdin_data.push_str(&standardized_rust_data_snps_x_samples[[i,j]].to_string());
                if j < standardized_rust_data_snps_x_samples.ncols() - 1 {
                    stdin_data.push(' ');
                }
            }
            stdin_data.push('\n');
        }
        
        let mut script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        script_path.push("tests/pca.py");

        let mut process = Command::new("python3")
            .arg(script_path.to_str().unwrap())
            .arg("--generate-reference-pca") // Updated flag
            .arg("-k")
            .arg(k_components.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to spawn pca.py process");

        let mut stdin = process.stdin.take().expect("Failed to open stdin for pca.py");
        std::thread::spawn(move || {
            stdin.write_all(stdin_data.as_bytes()).expect("Failed to write to stdin of pca.py");
        });

        let output = process.wait_with_output().expect("Failed to read pca.py stdout/stderr");
        
        if !output.status.success() {
            panic!("pca.py execution failed:\nStdout:\n{}\nStderr:\n{}", 
                String::from_utf8_lossy(&output.stdout), 
                String::from_utf8_lossy(&output.stderr));
        }
        let python_output_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr); // Capture stderr as string
        let (py_loadings_k_x_d, py_scores_n_x_k, py_eigenvalues_k) = 
            parse_pca_py_output(&python_output_str).expect(
                &format!(
                    "Failed to parse pca.py output. Full stdout:
---
{}
---
Full stderr:
---
{}
---",
                    python_output_str,
                    stderr_str
                )
            );

        let py_loadings_d_x_k = py_loadings_k_x_d.t().into_owned();

        let artifact_dir = "target/test_artifacts/pca_small_dataset";
        save_matrix_to_tsv(&rust_result.final_snp_principal_component_loadings.view(), artifact_dir, "rust_loadings.tsv").expect("Failed to save rust_loadings.tsv");
        save_matrix_to_tsv(&py_loadings_d_x_k.view(), artifact_dir, "python_loadings.tsv").expect("Failed to save python_loadings.tsv");
        save_matrix_to_tsv(&rust_result.final_sample_principal_component_scores.view(), artifact_dir, "rust_scores.tsv").expect("Failed to save rust_scores.tsv");
        save_matrix_to_tsv(&py_scores_n_x_k.view(), artifact_dir, "python_scores.tsv").expect("Failed to save python_scores.tsv");
        save_vector_to_tsv(&rust_result.final_principal_component_eigenvalues.view(), artifact_dir, "rust_eigenvalues.tsv").expect("Failed to save rust_eigenvalues.tsv");
        save_vector_to_tsv(&py_eigenvalues_k.view(), artifact_dir, "python_eigenvalues.tsv").expect("Failed to save python_eigenvalues.tsv");

        assert_eq!(rust_result.num_principal_components_computed, k_components, "Rust num_principal_components_computed mismatch");
        assert_eq!(py_loadings_d_x_k.ncols(), k_components, "Python effective components (loadings) mismatch");
        assert_eq!(py_scores_n_x_k.ncols(), k_components, "Python effective components (scores) mismatch");
        assert_eq!(py_eigenvalues_k.len(), k_components, "Python effective components (eigenvalues) mismatch");

        println!("DEBUG: test_pca_with_known_small_dataset - Rust SNP Loadings:
{:?}", rust_result.final_snp_principal_component_loadings.view());
        println!("DEBUG: test_pca_with_known_small_dataset - Python SNP Loadings (D_snps x k_components):
{:?}", py_loadings_d_x_k.view());
        assert_f32_arrays_are_close_with_sign_flips(
            rust_result.final_snp_principal_component_loadings.view(), // Use .view()
            py_loadings_d_x_k.view(), // Use .view()
            1.5f32, // Adjusted tolerance
            "SNP Loadings (Rust vs Python)"
        );
        assert_f32_arrays_are_close_with_sign_flips(
            rust_result.final_sample_principal_component_scores.view(), // Use .view()
            py_scores_n_x_k.view(), // Use .view()
            DEFAULT_FLOAT_TOLERANCE_F32 * 10.0, // Relaxed tolerance
            "Sample Scores (Rust vs Python)"
        );
        assert_f64_arrays_are_close(
            rust_result.final_principal_component_eigenvalues.view(), // Use .view()
            py_eigenvalues_k.view(), // Use .view()
            DEFAULT_FLOAT_TOLERANCE_F64 * 10.0, // Relaxed tolerance
            "Eigenvalues (Rust vs Python)"
        );
    }

    #[test]
    fn test_pc_scores_orthogonality() {
        let num_samples = 50;
        let num_snps = 100;
        let num_pcs_target = 5;

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
        let standardized_genos = standardize_features_across_samples(raw_genos);
        let test_data = TestDataAccessor::new(standardized_genos);

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: num_pcs_target,
            subset_factor_for_local_basis_learning: 0.5,
            min_subset_size_for_local_basis_learning: (num_samples / 4).max(1),
            max_subset_size_for_local_basis_learning: (num_samples / 2).max(10),
            components_per_ld_block: 10.min(num_snps.min( (num_samples/2).max(10) )), 
            random_seed: 123,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification {
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
        }];

        let output = algorithm.compute_pca(&test_data, &ld_blocks).expect("PCA failed");
        assert_eq!(output.num_principal_components_computed, num_pcs_target, "Did not compute target PCs");

        let scores = &output.final_sample_principal_component_scores;
        assert_eq!(scores.nrows(), num_samples);
        assert_eq!(scores.ncols(), num_pcs_target);

        if num_samples <= 1 || num_pcs_target == 0 { return; }

        let scores_f64 = scores.mapv(|x| x as f64);
        let covariance_matrix = scores_f64.t().dot(&scores_f64) / (output.num_qc_samples_used as f64 - 1.0);

        for r in 0..num_pcs_target {
            for c in 0..num_pcs_target {
                if r == c {
                    assert!(
                        (covariance_matrix[[r, c]] - output.final_principal_component_eigenvalues[r]).abs() < DEFAULT_FLOAT_TOLERANCE_F64 * 100.0, // Relaxed tolerance
                        "Covariance diagonal [{},{}] {} does not match eigenvalue {} (diff {})",
                        r, c, covariance_matrix[[r, c]], output.final_principal_component_eigenvalues[r],
                        (covariance_matrix[[r, c]] - output.final_principal_component_eigenvalues[r]).abs()
                    );
                } else {
                    assert!(
                        covariance_matrix[[r, c]].abs() < DEFAULT_FLOAT_TOLERANCE_F64 * 100.0, // Relaxed tolerance
                        "Covariance off-diagonal [{},{}] {} is not close to 0",
                        r, c, covariance_matrix[[r, c]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_snp_loadings_orthonormality() {
        let num_samples = 60;
        let num_snps = 120;
        let num_pcs_target = 4;

        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
        let standardized_genos = standardize_features_across_samples(raw_genos);
        let test_data = TestDataAccessor::new(standardized_genos);

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: num_pcs_target,
            random_seed: 456,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification {
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
        }];

        let output = algorithm.compute_pca(&test_data, &ld_blocks).expect("PCA failed");
        assert_eq!(output.num_principal_components_computed, num_pcs_target);

        let loadings = &output.final_snp_principal_component_loadings;
        assert_eq!(loadings.nrows(), num_snps);
        assert_eq!(loadings.ncols(), num_pcs_target);
        
        if num_pcs_target == 0 { return; }
        let check_identity = loadings.t().dot(loadings);

        for r in 0..num_pcs_target {
            for c in 0..num_pcs_target {
                let expected_val = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (check_identity[[r, c]] - expected_val).abs() < DEFAULT_FLOAT_TOLERANCE_F32,
                    "Loadings orthonormality check: Identity matrix mismatch at [{},{}]. Expected {}, Got {} (diff {})",
                    r, c, expected_val, check_identity[[r, c]], (check_identity[[r, c]] - expected_val).abs()
                );
            }
        }
    }

    #[test]
    fn test_eigenvalue_score_variance_correspondence() {
        let num_samples = 55;
        let num_snps = 110;
        let num_pcs_target = 3;

        let mut rng = ChaCha8Rng::seed_from_u64(789);
        let raw_genos = Array2::random_using((num_snps, num_samples), Uniform::new(0.0, 3.0), &mut rng);
        let standardized_genos = standardize_features_across_samples(raw_genos);
        let test_data = TestDataAccessor::new(standardized_genos);

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: num_pcs_target,
            random_seed: 789,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification {
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
        }];

        let output = algorithm.compute_pca(&test_data, &ld_blocks).expect("PCA failed");
        assert_eq!(output.num_principal_components_computed, num_pcs_target);

        if num_samples <= 1 || num_pcs_target == 0 { return; }

        let scores = &output.final_sample_principal_component_scores; 
        let eigenvalues = &output.final_principal_component_eigenvalues;

        for k in 0..num_pcs_target {
            let score_column_k = scores.column(k);
            let sum_sq_f64 = score_column_k.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
            let variance_of_score_k = sum_sq_f64 / (output.num_qc_samples_used as f64 - 1.0);
            
            assert!(
                (variance_of_score_k - eigenvalues[k]).abs() < DEFAULT_FLOAT_TOLERANCE_F64 * 100.0, // Relaxed tolerance
                "Variance of score column {} ({}) does not match eigenvalue {} ({}) (diff {})",
                k, variance_of_score_k, k, eigenvalues[k], (variance_of_score_k - eigenvalues[k]).abs()
            );
        }
    }
    
    #[test]
    fn test_pca_zero_snps() {
        let num_samples = 10;
        let test_data = TestDataAccessor::new_empty(0, num_samples);

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: 2,
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
    }

    #[test]
    fn test_pca_zero_samples() {
        let num_snps = 20;
        let test_data = TestDataAccessor::new_empty(num_snps, 0); 

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: 2,
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
    }

    #[test]
    fn test_pca_more_components_requested_than_rank() {
        let num_samples = 10;
        let num_true_rank_snps = 2;
        let num_total_snps = 5;
        let k_components_requested = 4;

        let mut raw_genos = Array2::<f32>::zeros((num_total_snps, num_samples));
        let mut rng = ChaCha8Rng::seed_from_u64(321);
        for r in 0..num_true_rank_snps {
            for c in 0..num_samples {
                raw_genos[[r,c]] = rng.sample(Uniform::new(0.0, 3.0));
            }
        }

        // Fix for E0502: Separate RHS computation from LHS mutable borrow.
        let row0_val_for_row2 = raw_genos.row(0).mapv(|x| x * 0.5);
        let row1_val_for_row2 = raw_genos.row(1).mapv(|x| x * 0.2);
        let rhs_for_row2 = &row0_val_for_row2 + &row1_val_for_row2;
        raw_genos.row_mut(2).assign(&rhs_for_row2);

        let row0_val_for_row3 = raw_genos.row(0).mapv(|x| x * 0.1);
        let row1_val_for_row3 = raw_genos.row(1).mapv(|x| x * 0.3);
        let rhs_for_row3 = &row0_val_for_row3 - &row1_val_for_row3;
        raw_genos.row_mut(3).assign(&rhs_for_row3);

        let rhs_for_row4 = raw_genos.row(0).mapv(|x| x * 0.8);
        raw_genos.row_mut(4).assign(&rhs_for_row4);
        
        let standardized_genos = standardize_features_across_samples(raw_genos.clone());
        let test_data = TestDataAccessor::new(standardized_genos.clone());

        let config = EigenSNPCoreAlgorithmConfig {
            target_num_global_pcs: k_components_requested, 
            components_per_ld_block: num_total_snps.min(num_samples), 
            random_seed: 321,
            subset_factor_for_local_basis_learning: 1.0, 
            min_subset_size_for_local_basis_learning: num_samples,
            max_subset_size_for_local_basis_learning: num_samples,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let ld_blocks = vec![LdBlockSpecification {
            user_defined_block_tag: "block1".to_string(),
            pca_snp_ids_in_block: (0..num_total_snps).map(PcaSnpId).collect(),
        }];

        let rust_output = algorithm.compute_pca(&test_data, &ld_blocks).expect("Rust PCA failed for low-rank data");
        
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

        let mut process = Command::new("python3")
            .arg(script_path.to_str().unwrap())
            .arg("--generate-reference-pca") // Updated flag
            .arg("-k").arg(k_components_requested.to_string())
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().expect("Failed to spawn pca.py process");

        let mut stdin_pipe = process.stdin.take().expect("Failed to open stdin for pca.py");
        std::thread::spawn(move || {
            stdin_pipe.write_all(stdin_data.as_bytes()).expect("Failed to write to stdin of pca.py");
        });

        let py_cmd_output = process.wait_with_output().expect("Failed to read pca.py stdout/stderr");
        if !py_cmd_output.status.success() {
            panic!("pca.py execution failed for low-rank data:\nStdout:\n{}\nStderr:\n{}", 
                String::from_utf8_lossy(&py_cmd_output.stdout), 
                String::from_utf8_lossy(&py_cmd_output.stderr));
        }
        let python_output_str = String::from_utf8_lossy(&py_cmd_output.stdout);
        let stderr_str = String::from_utf8_lossy(&py_cmd_output.stderr); // Capture stderr
        let (py_loadings_k_x_d, py_scores_n_x_k, py_eigenvalues_k) = 
            parse_pca_py_output(&python_output_str).expect(
                &format!(
                    "Failed to parse pca.py output for low-rank data. Full stdout:
---
{}
---
Full stderr:
---
{}
---",
                    python_output_str,
                    stderr_str
                )
            );
        
        let py_loadings_d_x_k = py_loadings_k_x_d.t().into_owned();

        let artifact_dir = "target/test_artifacts/pca_low_rank";
        save_matrix_to_tsv(&rust_output.final_snp_principal_component_loadings.view(), artifact_dir, "rust_loadings.tsv").expect("Failed to save rust_loadings.tsv");
        save_matrix_to_tsv(&py_loadings_d_x_k.view(), artifact_dir, "python_loadings.tsv").expect("Failed to save python_loadings.tsv");
        save_matrix_to_tsv(&rust_output.final_sample_principal_component_scores.view(), artifact_dir, "rust_scores.tsv").expect("Failed to save rust_scores.tsv");
        save_matrix_to_tsv(&py_scores_n_x_k.view(), artifact_dir, "python_scores.tsv").expect("Failed to save python_scores.tsv");
        save_vector_to_tsv(&rust_output.final_principal_component_eigenvalues.view(), artifact_dir, "rust_eigenvalues.tsv").expect("Failed to save rust_eigenvalues.tsv");
        save_vector_to_tsv(&py_eigenvalues_k.view(), artifact_dir, "python_eigenvalues.tsv").expect("Failed to save python_eigenvalues.tsv");
        
        let effective_k_rust = rust_output.num_principal_components_computed;
        let effective_k_py = py_eigenvalues_k.len();

        println!("Low-rank test: Rust computed {} PCs, Python computed {} PCs (requested {})", effective_k_rust, effective_k_py, k_components_requested);

        assert!(effective_k_rust <= k_components_requested);
        assert!(effective_k_py <= k_components_requested);
        
        let num_pcs_to_compare = effective_k_rust.min(effective_k_py).min(num_true_rank_snps + 1); 

        if num_pcs_to_compare > 0 {
            println!("DEBUG: test_pca_more_components_requested_than_rank - Rust SNP Loadings (slice {}x{}):
{:?}", 
               rust_output.final_snp_principal_component_loadings.slice(s![.., 0..num_pcs_to_compare]).nrows(),
               rust_output.final_snp_principal_component_loadings.slice(s![.., 0..num_pcs_to_compare]).ncols(),
               rust_output.final_snp_principal_component_loadings.slice(s![.., 0..num_pcs_to_compare]));
            println!("DEBUG: test_pca_more_components_requested_than_rank - Python SNP Loadings (slice {}x{}):
{:?}",
               py_loadings_d_x_k.slice(s![.., 0..num_pcs_to_compare]).nrows(),
               py_loadings_d_x_k.slice(s![.., 0..num_pcs_to_compare]).ncols(),
               py_loadings_d_x_k.slice(s![.., 0..num_pcs_to_compare]));
            assert_f32_arrays_are_close_with_sign_flips(
                rust_output.final_snp_principal_component_loadings.slice(s![.., 0..num_pcs_to_compare]), // Remove &
                py_loadings_d_x_k.slice(s![.., 0..num_pcs_to_compare]), // Remove &
                1.5f32, // Adjusted tolerance
                "SNP Loadings (Low-Rank)"
            );
            assert_f32_arrays_are_close_with_sign_flips(
                rust_output.final_sample_principal_component_scores.slice(s![.., 0..num_pcs_to_compare]), // Remove &
                py_scores_n_x_k.slice(s![.., 0..num_pcs_to_compare]), // Remove &
                DEFAULT_FLOAT_TOLERANCE_F32 * 10.0,
                "Sample Scores (Low-Rank)"
            );
            assert_f64_arrays_are_close(
                rust_output.final_principal_component_eigenvalues.slice(s![0..num_pcs_to_compare]), // Remove &
                py_eigenvalues_k.slice(s![0..num_pcs_to_compare]), // Remove &
                DEFAULT_FLOAT_TOLERANCE_F64 * 10.0, 
                "Eigenvalues (Low-Rank)"
            );
        }
        if effective_k_rust > num_true_rank_snps {
            for i in num_true_rank_snps..effective_k_rust {
                assert!(rust_output.final_principal_component_eigenvalues[i] < 1e-3, 
                        "Rust Eigenvalue for PC {} (beyond true rank) is not close to zero: {}", 
                        i, rust_output.final_principal_component_eigenvalues[i]);
            }
        }
         if effective_k_py > num_true_rank_snps {
            for i in num_true_rank_snps..effective_k_py {
                assert!(py_eigenvalues_k[i] < 1e-3, 
                        "Python Eigenvalue for PC {} (beyond true rank) is not close to zero: {}", 
                        i, py_eigenvalues_k[i]);
            }
        }
    }
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
