#![cfg(feature = "enable-eigensnp-diagnostics")]

use efficient_pca::diagnostics::FullPcaRunDetailedDiagnostics;
use efficient_pca::eigensnp::{
    EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, EigenSNPCoreOutput, LdBlockSpecification,
    PcaReadyGenotypeAccessor, PcaSnpId, QcSampleId, ThreadSafeStdError,
}; // Removed RsvdStepDetail, PerBlockLocalBasisDiagnostics

use ndarray::Array2; // Removed Array1, Axis
use rand::Rng; // For genotype data generation
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json;
use std::fs::File; // Removed self
use std::io::{BufReader, Write}; // Added BufReader
use std::path::Path; // For parsing JSON

// Attempt to import TestResultRecord and TEST_RESULTS
// This path might need adjustment if the actual module structure is different.
// Assuming eigensnp_tests module is at the crate root for now.
// If `efficient_pca::eigensnp_tests` is not a public module, or these items are not public,
// this will fail compilation and require adjustment in a later step.
#[cfg(test)]
use efficient_pca::eigensnp_tests::TestResultRecord;

// --- Mock Genotype Data Accessor ---
#[derive(Clone)]
struct MockGenotypeData {
    genotypes: Array2<f32>, // SNPs x Samples
    num_qc_samples: usize,
    num_pca_snps: usize,
}

impl PcaReadyGenotypeAccessor for MockGenotypeData {
    fn get_standardized_snp_sample_block(
        &self,
        snp_ids: &[PcaSnpId],
        sample_ids: &[QcSampleId],
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        if snp_ids.is_empty() || sample_ids.is_empty() {
            return Ok(Array2::zeros((snp_ids.len(), sample_ids.len())));
        }

        let mut output_block = Array2::zeros((snp_ids.len(), sample_ids.len()));
        for (i, snp_id) in snp_ids.iter().enumerate() {
            let snp_row = self.genotypes.row(snp_id.0);
            for (j, sample_id) in sample_ids.iter().enumerate() {
                output_block[[i, j]] = snp_row[sample_id.0];
            }
        }
        Ok(output_block)
    }

    fn num_pca_snps(&self) -> usize {
        self.num_pca_snps
    }

    fn num_qc_samples(&self) -> usize {
        self.num_qc_samples
    }
}

// --- Helper Function to Run Diagnostic Test ---
#[allow(dead_code)]
fn run_diagnostic_test_with_params(
    num_snps: usize,
    num_samples: usize,
    num_ld_blocks: usize,
    components_per_ld_block: usize,
    target_num_global_pcs: usize,
    local_rsvd_num_power_iterations: usize, // Added this to vary it
    global_pca_num_power_iterations: usize, // Added this for completeness
    refine_pass_count: usize,
    diagnostic_block_list_id_to_trace: Option<usize>,
    output_filename_suffix: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // Returns path to output file
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let genotypes = Array2::from_shape_fn((num_snps, num_samples), |_| rng.gen_range(0.0..=2.0));
    let mock_data_accessor = MockGenotypeData {
        genotypes: genotypes.clone(),
        num_qc_samples: num_samples,
        num_pca_snps: num_snps,
    };

    let mut ld_block_specs = Vec::new();
    if num_ld_blocks > 0 && num_snps > 0 {
        let snps_per_block = (num_snps as f32 / num_ld_blocks as f32).ceil() as usize;
        let mut current_snp_idx = 0;
        for i in 0..num_ld_blocks {
            let end_snp_idx = (current_snp_idx + snps_per_block).min(num_snps);
            if current_snp_idx >= end_snp_idx && i < num_ld_blocks - 1 {
                continue;
            } // Avoid empty blocks unless it's the last one potentially
            if current_snp_idx >= num_snps {
                break;
            }

            let pca_snp_ids_in_block: Vec<PcaSnpId> =
                (current_snp_idx..end_snp_idx).map(PcaSnpId).collect();

            if pca_snp_ids_in_block.is_empty() && current_snp_idx < num_snps {
                // If somehow an empty block is generated for non-last blocks, assign remaining to it if needed.
                // This logic can be simplified if num_snps is always cleanly divisible or last block takes all.
                // For now, just ensure non-empty if possible.
                if i == num_ld_blocks - 1 && current_snp_idx < num_snps {
                    ld_block_specs.push(LdBlockSpecification {
                        user_defined_block_tag: format!("block_{}", i),
                        pca_snp_ids_in_block: (current_snp_idx..num_snps).map(PcaSnpId).collect(),
                    });
                    #[allow(unused_assignments)]
                    {
                        current_snp_idx = num_snps;
                    } // all assigned
                    break;
                }
                // else skip empty block if not the last one.
            } else if !pca_snp_ids_in_block.is_empty() {
                ld_block_specs.push(LdBlockSpecification {
                    user_defined_block_tag: format!("block_{}", i),
                    pca_snp_ids_in_block,
                });
            }
            current_snp_idx = end_snp_idx;
        }
        // If after loop, not all SNPs assigned (e.g. num_ld_blocks = 0 but num_snps > 0), create one encompassing block
        if num_snps > 0 && ld_block_specs.is_empty() {
            ld_block_specs.push(LdBlockSpecification {
                user_defined_block_tag: "block_0_catch_all".to_string(),
                pca_snp_ids_in_block: (0..num_snps).map(PcaSnpId).collect(),
            });
        }
    }

    let config = EigenSNPCoreAlgorithmConfig {
        subset_factor_for_local_basis_learning: 0.5,
        min_subset_size_for_local_basis_learning: (num_samples / 2).max(10).min(num_samples),
        max_subset_size_for_local_basis_learning: num_samples,
        components_per_ld_block,
        target_num_global_pcs,
        global_pca_sketch_oversampling: 5
            .min(if target_num_global_pcs > 0 {
                target_num_global_pcs - 1
            } else {
                0
            })
            .max(1), // Ensure > 0 if k > 0
        global_pca_num_power_iterations, // Use passed param
        local_rsvd_sketch_oversampling: 5
            .min(if components_per_ld_block > 0 {
                components_per_ld_block - 1
            } else {
                0
            })
            .max(1), // Ensure > 0 if cpb > 0
        local_rsvd_num_power_iterations, // Use passed param
        random_seed: 123,
        snp_processing_strip_size: 500.min(num_snps).max(1),
        refine_pass_count,
        collect_diagnostics: true,
        diagnostic_block_list_id_to_trace,
        local_pcs_output_dir: None,
    };

    let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
    let snp_metadata: Vec<efficient_pca::eigensnp::PcaSnpMetadata> = (0..mock_data_accessor
        .num_pca_snps())
        .map(|i| efficient_pca::eigensnp::PcaSnpMetadata {
            id: std::sync::Arc::new(format!("snp_{}", i)),
            chr: std::sync::Arc::new("chr1".to_string()),
            pos: i as u64 * 1000 + 100000,
        })
        .collect();
    let pca_result_tuple = algorithm
        .compute_pca(&mock_data_accessor, &ld_block_specs, &snp_metadata)
        .map_err(|e| e as Box<dyn std::error::Error>)?;

    let _pca_output: EigenSNPCoreOutput = pca_result_tuple.0;
    let detailed_diagnostics: Option<FullPcaRunDetailedDiagnostics> = pca_result_tuple.1;

    let dir = Path::new("test_outputs");
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let filename = dir.join(format!(
        "diag_n{}_d{}_b{}_cpb{}_k{}_sr{}_locIter{}_globIter{}_{}.json",
        num_samples,
        num_snps,
        ld_block_specs.len(), // Use actual number of blocks
        components_per_ld_block,
        target_num_global_pcs,
        refine_pass_count,
        local_rsvd_num_power_iterations,
        global_pca_num_power_iterations,
        output_filename_suffix
    ));

    if let Some(diagnostics) = detailed_diagnostics {
        let mut file = File::create(&filename)?;
        let json_string = serde_json::to_string_pretty(&diagnostics)?;
        file.write_all(json_string.as_bytes())?;
        println!("Diagnostics saved to {}", filename.display());
        Ok(filename.to_string_lossy().into_owned())
    } else {
        Err(Box::from("No diagnostics were collected."))
    }
}

// --- Test Functions ---

#[test]
fn diag_li0_gi2() {
    let test_name = "diag_li0_gi2";
    let num_snps = 1000;
    let num_samples = 100;
    let num_ld_blocks = 1;
    let components_per_ld_block = 5;
    let target_num_global_pcs = 5;
    let local_rsvd_num_power_iterations = 0;
    let global_pca_num_power_iterations = 2; // Fixed for this series
    let refine_pass_count = 1; // Minimal passes
    let diagnostic_block_id_to_trace = Some(0);
    let suffix = "local0_global2";

    let mut record = TestResultRecord {
        test_name: test_name.to_string(),
        num_features_d: num_snps,
        num_samples_n: num_samples,
        num_components_k: target_num_global_pcs,
        num_ld_blocks_b: num_ld_blocks,
        components_per_block_c: components_per_ld_block,
        success: false, // Default to false
        outcome_details: String::new(),
        notes: String::new(),
        approx_peak_mem_mibs: None,  // Not measured here
        computation_time_secs: None, // Not measured here
    };

    match run_diagnostic_test_with_params(
        num_snps,
        num_samples,
        num_ld_blocks,
        components_per_ld_block,
        target_num_global_pcs,
        local_rsvd_num_power_iterations,
        global_pca_num_power_iterations,
        refine_pass_count,
        diagnostic_block_id_to_trace,
        suffix,
    ) {
        Ok(filepath) => {
            record.notes = format!("Diagnostics JSON saved to: {}", filepath);
            let file = File::open(filepath).expect("Failed to open diagnostic file.");
            let reader = BufReader::new(file);
            let diagnostics: FullPcaRunDetailedDiagnostics =
                serde_json::from_reader(reader).expect("Failed to parse diagnostics JSON.");

            // Extract a key metric, e.g., condition number from a specific rSVD step
            if let Some(first_block_diag) = diagnostics.per_block_diagnostics.get(0) {
                if let Some(rsvd_step) = first_block_diag
                    .rsvd_stages
                    .iter()
                    .find(|s| s.step_name == "ProjectedB_PreSVD")
                {
                    if let Some(cond_num) = rsvd_step.condition_number {
                        let key_metric_info =
                            format!("ProjectedB_PreSVD CondNum: {:.4e}", cond_num);
                        println!("{}: {}", test_name, key_metric_info);
                        record.notes.push_str(&format!(" | {}", key_metric_info));
                    }
                }
            }

            record.success = true;
            record.outcome_details = format!(
                "Ran with local_power_iter={}",
                local_rsvd_num_power_iterations
            );
        }
        Err(e) => {
            record.success = false;
            record.outcome_details = format!("Test failed: {:?}", e);
            record.notes = "Test execution failed, no JSON generated.".to_string();
            panic!("Test {} failed: {:?}", test_name, e);
        }
    }

    // Assuming TEST_RESULTS is a globally accessible, mutable static collection
    // This part is tricky and depends on how TEST_RESULTS is implemented (e.g., using ctor, lazy_static with Mutex)
    // For now, let's simulate adding to it if it were a simple static Vec (which isn't directly possible for mutation).
    // This will likely need to be adapted based on the actual implementation of TEST_RESULTS.
    // unsafe { TEST_RESULTS.push(record) }; // This is a placeholder for actual result recording
    println!("Test Record for {}: {:?}", test_name, record); // Print for now
}

// Placeholder for other tests (diag_li2_gi2, diag_li4_gi2) - to be added in subsequent steps.

#[test]
fn diag_li2_gi2() {
    let test_name = "diag_li2_gi2";
    let num_snps = 1000;
    let num_samples = 100;
    let num_ld_blocks = 1; // Test with 1 block for focused local rsvd diagnostics
    let components_per_ld_block = 5;
    let target_num_global_pcs = 5;
    let local_rsvd_num_power_iterations = 2; // Varied parameter
    let global_pca_num_power_iterations = 2;
    let refine_pass_count = 1;
    let diagnostic_block_id_to_trace = Some(0);
    let suffix = "local2_global2";

    let mut record = TestResultRecord {
        test_name: test_name.to_string(),
        num_features_d: num_snps,
        num_samples_n: num_samples,
        num_components_k: target_num_global_pcs,
        num_ld_blocks_b: num_ld_blocks,
        components_per_block_c: components_per_ld_block,
        success: false,
        outcome_details: String::new(),
        notes: String::new(),
        approx_peak_mem_mibs: None,
        computation_time_secs: None,
    };

    match run_diagnostic_test_with_params(
        num_snps,
        num_samples,
        num_ld_blocks,
        components_per_ld_block,
        target_num_global_pcs,
        local_rsvd_num_power_iterations,
        global_pca_num_power_iterations,
        refine_pass_count,
        diagnostic_block_id_to_trace,
        suffix,
    ) {
        Ok(filepath) => {
            record.notes = format!("Diagnostics JSON saved to: {}", filepath);
            let file = File::open(filepath).expect("Failed to open diagnostic file.");
            let reader = BufReader::new(file);
            let diagnostics: FullPcaRunDetailedDiagnostics =
                serde_json::from_reader(reader).expect("Failed to parse diagnostics JSON.");

            if let Some(first_block_diag) = diagnostics.per_block_diagnostics.get(0) {
                if let Some(rsvd_step) = first_block_diag
                    .rsvd_stages
                    .iter()
                    .find(|s| s.step_name == "ProjectedB_PreSVD")
                {
                    if let Some(cond_num) = rsvd_step.condition_number {
                        let key_metric_info =
                            format!("ProjectedB_PreSVD CondNum: {:.4e}", cond_num);
                        println!("{}: {}", test_name, key_metric_info);
                        record.notes.push_str(&format!(" | {}", key_metric_info));
                    }
                }
            }

            record.success = true;
            record.outcome_details = format!(
                "Ran with local_power_iter={}",
                local_rsvd_num_power_iterations
            );
        }
        Err(e) => {
            record.success = false;
            record.outcome_details = format!("Test failed: {:?}", e);
            record.notes = "Test execution failed, no JSON generated.".to_string();
            panic!("Test {} failed: {:?}", test_name, e);
        }
    }
    println!("Test Record for {}: {:?}", test_name, record);
}

#[test]
fn diag_li4_gi2() {
    let test_name = "diag_li4_gi2";
    let num_snps = 1000;
    let num_samples = 100;
    let num_ld_blocks = 1;
    let components_per_ld_block = 5;
    let target_num_global_pcs = 5;
    let local_rsvd_num_power_iterations = 4; // Varied parameter
    let global_pca_num_power_iterations = 2;
    let refine_pass_count = 1;
    let diagnostic_block_id_to_trace = Some(0);
    let suffix = "local4_global2";

    let mut record = TestResultRecord {
        test_name: test_name.to_string(),
        num_features_d: num_snps,
        num_samples_n: num_samples,
        num_components_k: target_num_global_pcs,
        num_ld_blocks_b: num_ld_blocks,
        components_per_block_c: components_per_ld_block,
        success: false,
        outcome_details: String::new(),
        notes: String::new(),
        approx_peak_mem_mibs: None,
        computation_time_secs: None,
    };

    match run_diagnostic_test_with_params(
        num_snps,
        num_samples,
        num_ld_blocks,
        components_per_ld_block,
        target_num_global_pcs,
        local_rsvd_num_power_iterations,
        global_pca_num_power_iterations,
        refine_pass_count,
        diagnostic_block_id_to_trace,
        suffix,
    ) {
        Ok(filepath) => {
            record.notes = format!("Diagnostics JSON saved to: {}", filepath);
            let file = File::open(filepath).expect("Failed to open diagnostic file.");
            let reader = BufReader::new(file);
            let diagnostics: FullPcaRunDetailedDiagnostics =
                serde_json::from_reader(reader).expect("Failed to parse diagnostics JSON.");

            if let Some(first_block_diag) = diagnostics.per_block_diagnostics.get(0) {
                if let Some(rsvd_step) = first_block_diag
                    .rsvd_stages
                    .iter()
                    .find(|s| s.step_name == "ProjectedB_PreSVD")
                {
                    if let Some(cond_num) = rsvd_step.condition_number {
                        let key_metric_info =
                            format!("ProjectedB_PreSVD CondNum: {:.4e}", cond_num);
                        println!("{}: {}", test_name, key_metric_info);
                        record.notes.push_str(&format!(" | {}", key_metric_info));
                    }
                }
            }
            record.success = true;
            record.outcome_details = format!(
                "Ran with local_power_iter={}",
                local_rsvd_num_power_iterations
            );
        }
        Err(e) => {
            record.success = false;
            record.outcome_details = format!("Test failed: {:?}", e);
            record.notes = "Test execution failed, no JSON generated.".to_string();
            panic!("Test {} failed: {:?}", test_name, e);
        }
    }
    println!("Test Record for {}: {:?}", test_name, record);
}

// Original placeholder, can be removed now or kept if other manual tests are needed.
#[test]
fn placeholder_diagnostic_test() {
    assert!(true);
}
