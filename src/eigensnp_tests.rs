// src/eigensnp_tests.rs
#![cfg(test)]
use super::eigensnp::*; // To import items from eigensnp.rs
use ndarray::{Array1, Array2, ArrayView2, s, Axis};
use crate::eigensnp::{PcaReadyGenotypeAccessor, QcSampleId, PcaSnpId, LdBlockSpecification, ThreadSafeStdError, RawCondensedFeatures, StandardizedCondensedFeatures, InitialSamplePcScores, EigenSNPCoreAlgorithmConfig, EigenSNPCoreAlgorithm, PerBlockLocalSnpBasis, LdBlockListId}; // Explicit imports
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap; // If needed for TestGenotypeData

struct TestGenotypeData {
    genotypes: Array2<f32>,
    num_pca_snps: usize,
    num_qc_samples: usize,
    // Optional: consider HashMaps if IDs are not contiguous
    // pca_snp_id_map: HashMap<PcaSnpId, usize>,
    // qc_sample_id_map: HashMap<QcSampleId, usize>,
}

impl TestGenotypeData {
    pub fn new(genotypes: Array2<f32>) -> Self {
        let num_snps = genotypes.nrows();
        let num_samples = genotypes.ncols();
        TestGenotypeData {
            genotypes,
            num_pca_snps: num_snps,
            num_qc_samples: num_samples,
            // pca_snp_id_map: (0..num_snps).map(|i| (PcaSnpId(i), i)).collect(), // Example if using map
            // qc_sample_id_map: (0..num_samples).map(|i| (QcSampleId(i), i)).collect(), // Example if using map
        }
    }
}

#[cfg(test)]
mod test_compute_pca_integration {
    use super::*; // Imports TestGenotypeData, EigenSNPCoreAlgorithmConfig, etc.
    use crate::eigensnp::{EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, LdBlockSpecification, PcaSnpId, QcSampleId, EigenSNPCoreOutput};
    use ndarray::{Array1, Array2, Axis};
    use float_cmp::assert_approx_eq;

    // Helper to create TestGenotypeData with simple, predictable data
    fn create_genotype_data_for_integration(num_snps: usize, num_samples: usize) -> TestGenotypeData {
        let mut genotypes = Array2::<f32>::zeros((num_snps, num_samples));
        if num_snps > 0 && num_samples > 0 {
            for i in 0..num_snps {
                for j in 0..num_samples {
                    genotypes[[i, j]] = ((i + j % 3) % 5) as f32 - 2.0; // Some varied data
                }
            }
        }
        TestGenotypeData::new(genotypes)
    }

    // Helper for basic LD block structure (one block per N SNPs)
    fn create_ld_blocks(total_snps: usize, snps_per_block: usize) -> Vec<LdBlockSpecification> {
        if total_snps == 0 || snps_per_block == 0 {
            return Vec::new();
        }
        (0..total_snps)
            .step_by(snps_per_block)
            .enumerate()
            .map(|(block_idx, start_snp_idx)| {
                let end_snp_idx = (start_snp_idx + snps_per_block).min(total_snps);
                LdBlockSpecification {
                    user_defined_block_tag: format!("block_{}", block_idx),
                    pca_snp_ids_in_block: (start_snp_idx..end_snp_idx).map(PcaSnpId).collect(),
                }
            })
            .filter(|block| !block.pca_snp_ids_in_block.is_empty())
            .collect()
    }

    // Default config for many tests
    fn get_default_test_config() -> EigenSNPCoreAlgorithmConfig {
        EigenSNPCoreAlgorithmConfig {
            subset_factor_for_local_basis_learning: 0.5, // Use a larger fraction for small test N
            min_subset_size_for_local_basis_learning: 2,  // Small min N_s
            max_subset_size_for_local_basis_learning: 100,
            components_per_ld_block: 2,
            target_num_global_pcs: 2,
            global_pca_sketch_oversampling: 2,
            global_pca_num_power_iterations: 1,
            random_seed: 42,
        }
    }
    
    fn assert_output_shapes(output: &EigenSNPCoreOutput, expected_snps: usize, expected_samples: usize, expected_pcs: usize) {
        assert_eq!(output.final_snp_principal_component_loadings.shape(), &[expected_snps, expected_pcs], "SNP Loadings shape mismatch");
        assert_eq!(output.final_sample_principal_component_scores.shape(), &[expected_samples, expected_pcs], "Sample Scores shape mismatch");
        assert_eq!(output.final_principal_component_eigenvalues.shape(), &[expected_pcs], "Eigenvalues shape mismatch");
        assert_eq!(output.num_pca_snps_used, expected_snps, "num_pca_snps_used mismatch");
        assert_eq!(output.num_qc_samples_used, expected_samples, "num_qc_samples_used mismatch");
        assert_eq!(output.num_principal_components_computed, expected_pcs, "num_principal_components_computed mismatch");
    }

    #[test]
    fn test_pca_basic_sanity_check() { // Test 1
        let config = get_default_test_config();
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let genotype_data = create_genotype_data_for_integration(10, 5); // 10 SNPs, 5 samples
        let ld_blocks = create_ld_blocks(10, 5); // 2 blocks
        
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        // N_s calculation for this test:
        // config.subset_factor_for_local_basis_learning = 0.5, num_total_qc_samples = 5
        // desired_subset_sample_count = (0.5 * 5.0).round() as usize = 3.
        // config.min_subset_size_for_local_basis_learning = 2, config.max_subset_size_for_local_basis_learning = 100.
        // actual_subset_sample_count (N_s) = desired_subset_sample_count (3)
        //                                   .max(min_subset_size_for_local_basis_learning=2) (becomes 3)
        //                                   .min(max_subset_size_for_local_basis_learning=100) (becomes 3)
        //                                   .min(num_total_qc_samples=5) (becomes 3). So, N_s = 3.
        // For a block of 5 SNPs (config.components_per_ld_block = 2):
        // cp_actual = min(components_per_ld_block=2, num_snps_in_block=5, N_s=3) = 2.
        // Total condensed features = 2 blocks * 2 cp_actual = 4.
        let expected_pcs = config.target_num_global_pcs // K=2
                            .min(genotype_data.num_pca_snps()) // D=10
                            .min(genotype_data.num_qc_samples()) // N=5
                            .min(config.components_per_ld_block * ld_blocks.len()); // total_condensed_features=4
                            // So min(2,10,5,4) = 2
        assert_output_shapes(&result, 10, 5, expected_pcs);
    }

    #[test]
    fn test_pca_target_pcs_zero_errors() { // Test 2
        let mut config = get_default_test_config();
        config.target_num_global_pcs = 0;
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_genotype_data_for_integration(10, 5);
        let ld_blocks = create_ld_blocks(10, 5);
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks);
        assert!(result.is_err(), "Expected error for target_num_global_pcs = 0");
    }

    #[test]
    fn test_pca_no_ld_blocks_with_snps_errors() { // Test 3
        let config = get_default_test_config();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_genotype_data_for_integration(10, 5); // Has SNPs
        let ld_blocks: Vec<LdBlockSpecification> = Vec::new(); // No blocks
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks);
        assert!(result.is_err(), "Expected error for no LD blocks with SNPs present");
    }
    
    #[test]
    fn test_pca_no_ld_blocks_no_snps_ok() { // Companion to Test 3
        let config = get_default_test_config();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_genotype_data_for_integration(0, 5); // No SNPs
        let ld_blocks: Vec<LdBlockSpecification> = Vec::new(); 
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        assert_output_shapes(&result, 0, 5, 0); // Expect 0 PCs
    }

    #[test]
    fn test_pca_zero_qc_samples() { // Test 4
        let config = get_default_test_config();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_genotype_data_for_integration(10, 0); // 0 samples
        let ld_blocks = create_ld_blocks(10, 5);
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        assert_output_shapes(&result, 10, 0, 0);
    }

    #[test]
    fn test_pca_zero_pca_snps() { // Test 5
        let config = get_default_test_config();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_genotype_data_for_integration(0, 10); // 0 SNPs
        let ld_blocks = create_ld_blocks(0, 5); // No blocks needed if no SNPs
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        assert_output_shapes(&result, 0, 10, 0);
    }
    
    #[test]
    fn test_pca_subset_ns_becomes_zero_errors_if_snps_in_blocks() { // Test 6
        let mut config = get_default_test_config();
        config.subset_factor_for_local_basis_learning = 0.001;
        config.min_subset_size_for_local_basis_learning = 0; 
        config.max_subset_size_for_local_basis_learning = 0; // Force N_s = 0
        
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_genotype_data_for_integration(10, 5); // 5 samples
        let ld_blocks = create_ld_blocks(10, 5); // Has SNPs
        
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks);
        // This should error because N_s will be 0, but blocks have SNPs.
        assert!(result.is_err(), "Expected error when N_s is 0 but blocks have SNPs");
    }

    #[test]
    fn test_pca_num_components_per_block_becomes_zero() { // Test 7
        let mut config = get_default_test_config();
        config.components_per_ld_block = 0; // Request 0 components per block
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let genotype_data = create_genotype_data_for_integration(10, 5);
        let ld_blocks = create_ld_blocks(10, 5);
        
        // If c_p = 0, then local bases are empty, condensed features are empty.
        // This should lead to 0 global PCs.
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        assert_output_shapes(&result, 10, 5, 0);
    }
    
    #[test]
    fn test_pca_total_condensed_features_is_zero() { // Test 8
        // This is similar to Test 7, ensure components_per_ld_block leads to 0 total condensed features
        let mut config = get_default_test_config();
        config.components_per_ld_block = 1; // Small c_p
        
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let genotype_data = create_genotype_data_for_integration(10, 5); // Has SNPs and Samples
        let ld_blocks = vec![
            LdBlockSpecification { user_defined_block_tag: "b1".into(), pca_snp_ids_in_block: vec![] },
            LdBlockSpecification { user_defined_block_tag: "b2".into(), pca_snp_ids_in_block: vec![] }
        ]; // All blocks have 0 SNPs
         // This means local bases will have 0 columns (since num_snps_in_block = 0 for each). Total condensed features = 0.
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        assert_output_shapes(&result, 10, 5, 0);
    }

    #[test]
    fn test_pca_initial_pca_yields_zero_components() { // Test 9
        let mut config = get_default_test_config();
        config.min_subset_size_for_local_basis_learning = 1; 
        config.max_subset_size_for_local_basis_learning = 1; // Force N_s = 1
        config.subset_factor_for_local_basis_learning = 0.99; // ensure factor doesn't make it 0 if N_samples=1
        
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        // Use data that might lead to zero variance in blocks with N_s=1
        // If all subset samples are identical for the SNPs in a block, cov matrix is zero.
        let mut genotypes = Array2::<f32>::zeros((10,3)); // 10 SNPs, 3 samples
        // Make all samples identical for first few SNPs in a block
        // N_s will be 1. Let's say sample 0 is chosen.
        // Genotypes for sample 0 for SNPs 0-4 are all 1.0
        for i in 0..5 { genotypes[[i,0]] = 1.0; } 
        // To ensure the *standardized* data for this sample for these SNPs is zero (or constant),
        // we need to control global mu/sigma. TestGenotypeData doesn't do global standardization.
        // However, if X_subset (1 col) is constant for a block, X_subset . X_subset.t() will have rank 1 or 0.
        // If it's rank 0 (all zeros), eigh gives zero eigenvectors.
        // If this happens for all blocks, A_eigen can be all zero.
        // Let's make the first block's subset data all zero.
        // If PcaSnpId(0)..PcaSnpId(4) are used with QcSampleId(0) (assuming it's picked for N_s=1)
        // and those genotypes are all identical (e.g. 0.0), then standardized data for that sample for those SNPs will be 0.
        // This makes the covariance matrix 0, local basis 0, projection 0.
        // If all blocks are like this, A_eigen is 0 -> A_eigen_std is 0 -> U_scores* is 0.
        for i in 0..10 { genotypes[[i,0]] = 0.0; } // Make sample 0 all zeros
                                                   // N_s will be 1 (sample 0 if seed is fixed and it's first pick, or just any single sample)
                                                   // If sample 0 is picked, all its standardized data for a block is 0.
                                                   // Then local basis for that block is 0.
                                                   // Then projection for all samples onto that basis is 0.
                                                   // So A_eigen_star would be all 0s. -> A_eigen_std_star all 0s.
                                                   // RSVD on all 0s matrix -> 0 scores.
        
        let genotype_data = TestGenotypeData::new(genotypes);
        let ld_blocks = create_ld_blocks(10, 5);

        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        assert_eq!(result.num_principal_components_computed, 0, "Expected 0 PCs if initial PCA fails");
        assert_output_shapes(&result, 10, 3, 0);
    }

    #[test]
    fn test_pca_refined_loadings_yield_zero_components() { // Test 10
        let config = get_default_test_config();
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        // Force genotype data to be all zeros
        let zero_genotypes = TestGenotypeData::new(Array2::<f32>::zeros((10,5)));
        let ld_blocks = create_ld_blocks(10,5);

        // If X is zero, A_eigen will be zero, A_eigen_std will be zero.
        // Initial scores U* from RSVD on zero matrix will be zero.
        // Then refined loadings V_final = QR(X^T U*) = QR(0) will be zero.
        let result = algorithm.compute_pca(&zero_genotypes, &ld_blocks).unwrap();
        assert_output_shapes(&result, 10, 5, 0);
    }
    
    #[test]
    fn test_pca_larger_n_d_multiple_blocks() { // Test 11
        let mut config = get_default_test_config();
        config.target_num_global_pcs = 5;
        config.components_per_ld_block = 4;
        config.min_subset_size_for_local_basis_learning = 10; // N_s will be at least 10
        config.subset_factor_for_local_basis_learning = 0.3; // 0.3 * 30 = 9. So N_s clamped to 10.

        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let num_snps = 50;
        let num_samples = 30;
        let genotype_data = create_genotype_data_for_integration(num_snps, num_samples);
        let ld_blocks = create_ld_blocks(num_snps, 10); // 5 blocks, 10 SNPs each
        // N_s = max(min_N_s=10, round(0.3*30=9)) = 10. Clamped by max_N_s=100. N_s=10.
        // For a block of 10 SNPs, cp_actual = min(cp=4, #SNPs=10, N_s=10) = 4.
        // Total condensed features = 5 blocks * 4 cp_actual = 20.
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        
        let expected_pcs = config.target_num_global_pcs // K=5
                            .min(num_snps) // D=50
                            .min(num_samples) // N=30
                            .min(config.components_per_ld_block * ld_blocks.len()); // total_condensed_features=20
                            // So min(5,50,30,20) = 5
        assert_output_shapes(&result, num_snps, num_samples, expected_pcs);
    }

    #[test]
    fn test_pca_cp_high_capped_by_snps_samples() { // Test 12
        let mut config = get_default_test_config();
        config.components_per_ld_block = 100; // Very high c_p
        config.target_num_global_pcs = 3;

        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let num_snps = 10; 
        let num_samples = 8; 
        let genotype_data = create_genotype_data_for_integration(num_snps, num_samples);
        let ld_blocks = create_ld_blocks(num_snps, 5); // 2 blocks, 5 SNPs each
        // N_s from config: (0.5 * 8 samples).round() = 4. Max(4, min_N_s=2) = 4. Min(4, max_N_s=100) = 4. So N_s = 4.
        // For a block of 5 SNPs, cp_actual = min(cp=100, #SNPs=5, N_s=4) = 4.
        // Total condensed features = 2 blocks * 4 cp_actual = 8.
                                                       
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        // Expected PCs = min(target_K=3, total_condensed_features=8, N=8, D=10) = 3
        assert_output_shapes(&result, num_snps, num_samples, config.target_num_global_pcs);
    }
    
    #[test]
    fn test_pca_ns_clamped_by_min_subset_size() { // Test 13
        let mut config = get_default_test_config();
        config.subset_factor_for_local_basis_learning = 0.01; 
        config.min_subset_size_for_local_basis_learning = 6; 
        config.max_subset_size_for_local_basis_learning = 20;
        config.target_num_global_pcs = 2;
        config.components_per_ld_block = 3;

        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let num_snps = 15;
        let num_samples = 10; 
        let genotype_data = create_genotype_data_for_integration(num_snps, num_samples);
        let ld_blocks = create_ld_blocks(num_snps, 7); // 3 blocks: (0..7), (7..14), (14..15)
        // N_s from config: (0.01 * 10 samples).round() = 0. Max(0, min_N_s=6) = 6. Min(6, max_N_s=20) = 6. So N_s = 6.
        // Block 1 (7 SNPs): cp_actual = min(cp=3, #SNPs=7, N_s=6) = 3.
        // Block 2 (7 SNPs): cp_actual = min(cp=3, #SNPs=7, N_s=6) = 3.
        // Block 3 (1 SNP): cp_actual = min(cp=3, #SNPs=1, N_s=6) = 1.
        // Total condensed features = 3 + 3 + 1 = 7.
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        // Expected PCs = min(target_K=2, total_condensed_features=7, N=10, D=15) = 2
        assert_output_shapes(&result, num_snps, num_samples, config.target_num_global_pcs);
    }

    #[test]
    fn test_pca_ns_clamped_by_max_subset_size() { // Test 14
        let mut config = get_default_test_config();
        config.subset_factor_for_local_basis_learning = 0.9; 
        config.min_subset_size_for_local_basis_learning = 2;
        config.max_subset_size_for_local_basis_learning = 5; 
        config.target_num_global_pcs = 2;
        config.components_per_ld_block = 3;

        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        let num_snps = 15;
        let num_samples = 10;
        let genotype_data = create_genotype_data_for_integration(num_snps, num_samples);
        let ld_blocks = create_ld_blocks(num_snps, 8); // 2 blocks: (0..8), (8..15)
        // N_s from config: (0.9 * 10 samples).round() = 9. Max(9, min_N_s=2) = 9. Min(9, max_N_s=5) = 5. So N_s = 5.
        // Block 1 (8 SNPs): cp_actual = min(cp=3, #SNPs=8, N_s=5) = 3.
        // Block 2 (7 SNPs): cp_actual = min(cp=3, #SNPs=7, N_s=5) = 3.
        // Total condensed features = 3 + 3 = 6.
        let result = algorithm.compute_pca(&genotype_data, &ld_blocks).unwrap();
        // Expected PCs = min(target_K=2, total_condensed_features=6, N=10, D=15) = 2
        assert_output_shapes(&result, num_snps, num_samples, config.target_num_global_pcs);
    }

    #[test]
    fn test_pca_different_random_seeds_yield_consistent_shapes() { // Test 15
        let mut config1 = get_default_test_config();
        config1.random_seed = 100;
        let algorithm1 = EigenSNPCoreAlgorithm::new(config1.clone());

        let mut config2 = get_default_test_config();
        config2.random_seed = 200;
        let algorithm2 = EigenSNPCoreAlgorithm::new(config2.clone());

        let genotype_data = create_genotype_data_for_integration(12, 7);
        let ld_blocks = create_ld_blocks(12, 4); // 3 blocks
        
        let result1 = algorithm1.compute_pca(&genotype_data, &ld_blocks).unwrap();
        let result2 = algorithm2.compute_pca(&genotype_data, &ld_blocks).unwrap();
        
        // N_s from config1/2: (0.5 * 7 samples).round() = 4. Max(4, min_N_s=2)=4. Min(4, max_N_s=100)=4. So N_s=4.
        // For a block of 4 SNPs, cp_actual = min(cp=2, #SNPs=4, N_s=4) = 2.
        // Total condensed features = 3 blocks * 2 cp_actual = 6.
        let expected_pcs = config1.target_num_global_pcs // K=2
                            .min(12) // D
                            .min(7)  // N
                            .min(config1.components_per_ld_block * ld_blocks.len()); // total_condensed_features=6
                            // min(2,12,7,6) = 2
        assert_output_shapes(&result1, 12, 7, expected_pcs);
        assert_output_shapes(&result2, 12, 7, expected_pcs);
        
        // Values might differ due to RSVD and subset selection, but shapes should be identical.
        // A more robust test might check if eigenvalues are roughly similar if data is large enough,
        // but for small data, differences can be large. Here, we primarily test that it runs and shapes are okay.
    }
}

#[cfg(test)]
mod test_compute_final_scores_and_eigenvalues {
    use super::*; // Imports TestGenotypeData, EigenSNPCoreAlgorithmConfig, etc.
    use crate::eigensnp::{EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, PcaSnpId, QcSampleId};
    use ndarray::{arr1, arr2, array, Array1, Array2, ArrayView2, Axis};
    use float_cmp::assert_approx_eq;

    // Helper to create TestGenotypeData with patterned data
    fn create_test_genotype_accessor(num_snps: usize, num_samples: usize, scale: f32) -> TestGenotypeData {
        let mut genotypes = Array2::<f32>::zeros((num_snps, num_samples));
        for i in 0..num_snps {
            for j in 0..num_samples {
                genotypes[[i, j]] = ((i % 5) as f32 - 2.0 + (j % 3) as f32 * 0.5) * scale;
            }
        }
        TestGenotypeData::new(genotypes)
    }
    
    // Helper to use a direct array for genotype_accessor in specific tests
    fn create_test_genotype_accessor_from_array(data: Array2<f32>) -> TestGenotypeData {
        TestGenotypeData::new(data)
    }

    #[test]
    fn test_final_scores_basic_calculation() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);

        let num_total_pca_snps = 4;
        let num_total_qc_samples = 3;
        let num_final_computed_pcs = 2;

        let genotype_data_arr = arr2(&[[1.0, 2.0, 0.0], // SNP0
                                       [0.0, 1.0, 3.0], // SNP1
                                       [2.0, 0.0, 1.0], // SNP2
                                       [1.0, 1.0, 1.0]]);// SNP3
        let genotype_accessor = create_test_genotype_accessor_from_array(genotype_data_arr.clone());
        
        let snp_loadings_arr = arr2(&[[0.5, 0.5],
                                      [0.5, -0.5],
                                      [0.5, 0.5],
                                      [0.5, -0.5]]);
        
        let (final_scores, final_eigenvalues) = algorithm.compute_final_scores_and_eigenvalues(
            &genotype_accessor,
            &snp_loadings_arr.view(),
            num_total_qc_samples
        ).unwrap();

        let expected_scores = genotype_data_arr.t().dot(&snp_loadings_arr);
        
        assert_eq!(final_scores.shape(), &[num_total_qc_samples, num_final_computed_pcs]);
        assert_eq!(final_eigenvalues.shape(), &[num_final_computed_pcs]);

        for r in 0..num_total_qc_samples {
            for c in 0..num_final_computed_pcs {
                assert_approx_eq!(f32, final_scores[[r,c]], expected_scores[[r,c]], epsilon=1e-6);
            }
        }

        if num_total_qc_samples > 1 {
            for k in 0..num_final_computed_pcs {
                let score_col_k = expected_scores.column(k);
                // Eigenvalue in code is sum_of_squares / (N-1)
                let sum_sq_k: f64 = score_col_k.iter().map(|&s| (s as f64).powi(2)).sum();
                let expected_eigenval_k = sum_sq_k / ((num_total_qc_samples - 1) as f64);
                assert_approx_eq!(f64, final_eigenvalues[k], expected_eigenval_k, epsilon=1e-6);
            }
        } else if num_total_qc_samples == 1 {
             for k_val in final_eigenvalues.iter() { // Iterate over eigenvalues for N=1 case
                assert_approx_eq!(f64, *k_val, 0.0, epsilon=1e-6);
             }
        }
    }

    #[test]
    fn test_final_scores_no_pcs() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let num_total_qc_samples = 5;
        let genotype_accessor = create_test_genotype_accessor(10, num_total_qc_samples, 1.0);
        let snp_loadings_arr = Array2::<f32>::zeros((10, 0)); // 0 PCs

        let (final_scores, final_eigenvalues) = algorithm.compute_final_scores_and_eigenvalues(
            &genotype_accessor,
            &snp_loadings_arr.view(),
            num_total_qc_samples
        ).unwrap();

        assert_eq!(final_scores.shape(), &[num_total_qc_samples, 0]);
        assert_eq!(final_eigenvalues.shape(), &[0]);
    }

    #[test]
    fn test_final_scores_no_snps() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let num_total_qc_samples = 5;
        let num_final_computed_pcs = 2;
        let genotype_accessor = create_test_genotype_accessor(0, num_total_qc_samples, 1.0);
        let snp_loadings_arr = Array2::<f32>::zeros((0, num_final_computed_pcs));

        let (final_scores, final_eigenvalues) = algorithm.compute_final_scores_and_eigenvalues(
            &genotype_accessor,
            &snp_loadings_arr.view(),
            num_total_qc_samples
        ).unwrap();
        
        assert_eq!(final_scores.shape(), &[num_total_qc_samples, num_final_computed_pcs]);
        for val in final_scores.iter() {
            assert_approx_eq!(f32, *val, 0.0, epsilon=1e-7);
        }
        for val in final_eigenvalues.iter() {
            assert_approx_eq!(f64, *val, 0.0, epsilon=1e-7);
        }
    }
    
    #[test]
    fn test_final_scores_single_sample() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);

        let num_total_pca_snps = 4;
        let num_total_qc_samples = 1; 
        let num_final_computed_pcs = 2;

        let genotype_data_arr = arr2(&[[1.0], [0.0], [2.0], [1.0]]); 
        let genotype_accessor = create_test_genotype_accessor_from_array(genotype_data_arr.clone());
        let snp_loadings_arr = arr2(&[[0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0.5, -0.5]]);
        
        let (final_scores, final_eigenvalues) = algorithm.compute_final_scores_and_eigenvalues(
            &genotype_accessor,
            &snp_loadings_arr.view(),
            num_total_qc_samples
        ).unwrap();

        let expected_scores = genotype_data_arr.t().dot(&snp_loadings_arr);
        assert_eq!(final_scores.shape(), &[num_total_qc_samples, num_final_computed_pcs]);
        assert_eq!(final_eigenvalues.shape(), &[num_final_computed_pcs]);
        for r in 0..num_total_qc_samples {
            for c in 0..num_final_computed_pcs {
                assert_approx_eq!(f32, final_scores[[r,c]], expected_scores[[r,c]], epsilon=1e-6);
            }
        }
        for k_val in final_eigenvalues.iter() {
            assert_approx_eq!(f64, *k_val, 0.0, epsilon=1e-6);
        }
    }
    
    #[test]
    fn test_final_scores_zero_samples() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let num_total_pca_snps = 4;
        let num_total_qc_samples = 0; 
        let num_final_computed_pcs = 2;

        let genotype_accessor = create_test_genotype_accessor(num_total_pca_snps, num_total_qc_samples, 1.0);
        // Loadings matrix would have num_total_pca_snps rows, num_final_computed_pcs columns
        let snp_loadings_arr = Array2::<f32>::zeros((num_total_pca_snps, num_final_computed_pcs)); 
        
        let (final_scores, final_eigenvalues) = algorithm.compute_final_scores_and_eigenvalues(
            &genotype_accessor,
            &snp_loadings_arr.view(),
            num_total_qc_samples
        ).unwrap();

        assert_eq!(final_scores.shape(), &[0, num_final_computed_pcs]); 
        assert_eq!(final_eigenvalues.shape(), &[num_final_computed_pcs]); 
        for val in final_eigenvalues.iter() {
            assert_approx_eq!(f64, *val, 0.0, epsilon=1e-7);
        }
    }
}

#[cfg(test)]
mod test_compute_refined_snp_loadings {
    use super::*; // Imports TestGenotypeData, EigenSNPCoreAlgorithmConfig, etc.
    use crate::eigensnp::{EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, InitialSamplePcScores, PcaSnpId, QcSampleId};
    use ndarray::{arr2, Array2, Axis};
    use float_cmp::assert_approx_eq;

    fn create_test_genotype_accessor(num_snps: usize, num_samples: usize) -> TestGenotypeData {
        let mut genotypes = Array2::<f32>::zeros((num_snps, num_samples));
        // Fill with some patterned data
        for i in 0..num_snps {
            for j in 0..num_samples {
                genotypes[[i, j]] = (i as f32 * 0.5 + j as f32 * 0.1) % 5.0;
            }
        }
        TestGenotypeData::new(genotypes)
    }

    #[test]
    fn test_refined_loadings_basic_shape() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);

        let num_total_pca_snps = 10;
        let num_qc_samples = 5;
        let num_initial_pcs = 3;

        let genotype_data = create_test_genotype_accessor(num_total_pca_snps, num_qc_samples);
        
        // Mock initial sample PC scores (N x K_initial)
        let mut initial_scores_data = Array2::<f32>::zeros((num_qc_samples, num_initial_pcs));
        for i in 0..num_qc_samples {
            for k in 0..num_initial_pcs {
                initial_scores_data[[i, k]] = (i + k) as f32;
            }
        }
        // Ensure initial_scores_data columns are not all zero or linearly dependent for QR to work as expected
        if num_initial_pcs > 1 && num_qc_samples > 1 {
            initial_scores_data[[0,0]] = 1.0; // Make them a bit more varied
            initial_scores_data[[1,1]] = 1.0;
            if num_initial_pcs > 2 && num_qc_samples > 2 {
                 initial_scores_data[[2,2]] = 1.0;
            }
        }


        let initial_sample_pc_scores = InitialSamplePcScores { scores: initial_scores_data };

        let result_loadings = algorithm.compute_refined_snp_loadings(
            &genotype_data,
            &initial_sample_pc_scores
        ).unwrap();

        // Expected shape: (num_total_pca_snps, num_initial_pcs after QR, which should be num_initial_pcs if rank is full)
        // The QR decomposition step means the number of columns in the output loadings
        // will be min(num_total_pca_snps, num_initial_pcs), assuming the matrix is full rank.
        let expected_cols = num_initial_pcs.min(num_total_pca_snps);
        assert_eq!(result_loadings.shape(), &[num_total_pca_snps, expected_cols]);
        
        // Check for orthogonality of columns due to QR
        if expected_cols > 1 {
            let col0 = result_loadings.column(0);
            let col1 = result_loadings.column(1);
            assert_approx_eq!(f32, col0.dot(&col1), 0.0, epsilon = 1e-5); // Orthogonal columns
             // Check for unit norm of columns (property of Q from QR)
            assert_approx_eq!(f32, col0.norm_l2(), 1.0, epsilon = 1e-5);
            assert_approx_eq!(f32, col1.norm_l2(), 1.0, epsilon = 1e-5);
        } else if expected_cols == 1 {
            assert_approx_eq!(f32, result_loadings.column(0).norm_l2(), 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_refined_loadings_no_initial_pcs() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let num_total_pca_snps = 10;
        let num_qc_samples = 5;
        let genotype_data = create_test_genotype_accessor(num_total_pca_snps, num_qc_samples);
        
        let initial_sample_pc_scores = InitialSamplePcScores { scores: Array2::zeros((num_qc_samples, 0)) }; // 0 initial PCs

        let result_loadings = algorithm.compute_refined_snp_loadings(
            &genotype_data,
            &initial_sample_pc_scores
        ).unwrap();
        
        assert_eq!(result_loadings.shape(), &[num_total_pca_snps, 0]);
    }

    #[test]
    fn test_refined_loadings_no_pca_snps() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let num_total_pca_snps = 0;
        let num_qc_samples = 5;
        let num_initial_pcs = 3;
        let genotype_data = create_test_genotype_accessor(num_total_pca_snps, num_qc_samples);
        
        let initial_scores_data = Array2::<f32>::zeros((num_qc_samples, num_initial_pcs));
        let initial_sample_pc_scores = InitialSamplePcScores { scores: initial_scores_data };

        let result_loadings = algorithm.compute_refined_snp_loadings(
            &genotype_data,
            &initial_sample_pc_scores
        ).unwrap();
        
        // num_total_pca_snps is 0, so output should have 0 rows.
        // Number of columns is min(0, num_initial_pcs) = 0
        assert_eq!(result_loadings.shape(), &[0, 0]);
    }
    
    #[test]
    fn test_refined_loadings_more_snps_than_samples() {
        // This tests if the dot product X^T U_scores* and subsequent QR handles D > N case for X
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);

        let num_total_pca_snps = 10; // More SNPs
        let num_qc_samples = 3;    // Fewer samples
        let num_initial_pcs = 2;

        let genotype_data = create_test_genotype_accessor(num_total_pca_snps, num_qc_samples);
        let mut initial_scores_data = Array2::<f32>::zeros((num_qc_samples, num_initial_pcs));
        for i in 0..num_qc_samples {
            for k in 0..num_initial_pcs { initial_scores_data[[i, k]] = (i + k * 2) as f32; }
        }
         // Ensure initial_scores_data columns are not all zero or linearly dependent for QR to work as expected
        if num_initial_pcs > 1 && num_qc_samples > 1 {
            initial_scores_data[[0,0]] = 1.0; 
            initial_scores_data[[1,1]] = 1.0;
             if num_initial_pcs > 2 && num_qc_samples > 2 {
                 initial_scores_data[[2,2]] = 1.0;
            }
        }


        let initial_sample_pc_scores = InitialSamplePcScores { scores: initial_scores_data };

        let result_loadings = algorithm.compute_refined_snp_loadings(
            &genotype_data,
            &initial_sample_pc_scores
        ).unwrap();
        
        let expected_cols = num_initial_pcs.min(num_total_pca_snps); // Should be min(num_initial_pcs, num_total_pca_snps, num_qc_samples) effectively from X_T U_scores*
                                                                    // The matrix for QR is (SNPs x InitialPCs). Its rank is min(SNPs, InitialPCs, Samples)
                                                                    // For QR, rank is min(rows, cols) of that matrix. So min(num_total_pca_snps, num_initial_pcs)
        assert_eq!(result_loadings.shape(), &[num_total_pca_snps, expected_cols]);
        if expected_cols > 1 {
            assert_approx_eq!(f32, result_loadings.column(0).dot(&result_loadings.column(1)), 0.0, epsilon = 1e-5);
            assert_approx_eq!(f32, result_loadings.column(0).norm_l2(), 1.0, epsilon = 1e-5);
        } else if expected_cols == 1 {
             assert_approx_eq!(f32, result_loadings.column(0).norm_l2(), 1.0, epsilon = 1e-5);
        }
    }
}

#[cfg(test)]
mod test_perform_randomized_svd_for_scores {
    use super::*; // Imports TestGenotypeData, EigenSNPCoreAlgorithmConfig, etc.
    use crate::eigensnp::EigenSNPCoreAlgorithm; // To access the now pub(crate) static method
    use ndarray::{arr2, Array2, ArrayView2};
    use float_cmp::assert_approx_eq;

    // Helper to create a simple matrix for testing
    fn create_matrix(rows: usize, cols: usize, start_val: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| (r * cols + c) as f32 + start_val)
    }

    #[test]
    fn test_rsvd_basic_case() {
        // A simple matrix (features_by_samples)
        // More features than samples, more samples than components
        let matrix_fs = arr2(&[[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0],
                               [10.0, 11.0, 12.0]]); // 4 features, 3 samples
        let num_components_target_k = 2;
        let sketch_oversampling_count = 1; // sketch_dim = 3 (min(4,3))
        let num_power_iterations = 1;
        let random_seed = 42;

        let result_scores_ns_by_k = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(),
            num_components_target_k,
            sketch_oversampling_count,
            num_power_iterations,
            random_seed
        ).unwrap();

        assert_eq!(result_scores_ns_by_k.shape(), &[3, num_components_target_k]); // num_samples x k
        // Further checks would involve comparing against a known SVD result,
        // but RSVD is approximate. For unit tests, shape and basic properties are key.
        // Check that columns are orthogonal (or close to it for RSVD)
        if num_components_target_k > 1 && result_scores_ns_by_k.ncols() > 1 { // ensure there are at least 2 columns to compare
            let col0 = result_scores_ns_by_k.column(0);
            let col1 = result_scores_ns_by_k.column(1);
            let dot_product = col0.dot(&col1);
            // For true SVD U matrix, dot product of columns is 0. RSVD might be approximate.
            // This also depends on whether the output scores are U or V*S from SVD of B.
            // The function returns V_B.T (right singular vectors of projected B), which should be orthogonal.
            assert_approx_eq!(f32, dot_product, 0.0, epsilon = 1e-3); // Allow some tolerance
        }
    }

    #[test]
    fn test_rsvd_target_k_zero() {
        let matrix_fs = create_matrix(5, 4, 1.0); // 5 features, 4 samples
        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), 0, 5, 1, 42
        ).unwrap();
        assert_eq!(result.shape(), &[4, 0]); // num_samples x 0
    }

    #[test]
    fn test_rsvd_empty_input_matrix_features() {
        let matrix_fs = Array2::<f32>::zeros((0, 4)); // 0 features, 4 samples
        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), 2, 5, 1, 42
        ).unwrap();
        assert_eq!(result.shape(), &[4, 0]); // num_samples x 0
    }

    #[test]
    fn test_rsvd_empty_input_matrix_samples() {
        let matrix_fs = Array2::<f32>::zeros((5, 0)); // 5 features, 0 samples
        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), 2, 5, 1, 42
        ).unwrap();
        assert_eq!(result.shape(), &[0, 0]); // 0 samples x 0 (or 0 samples x k_actual which is 0)
    }
    
    #[test]
    fn test_rsvd_sketch_dim_becomes_one() { // Renamed from sketch_dim_becomes_zero for clarity
        // Test case where k + oversampling > min(features, samples), so sketch_dimension = min(features, samples)
        // If min(features, samples) is very small.
        let matrix_fs = arr2(&[[1.0], [2.0]]); // 2 features, 1 sample
        let num_components_target_k = 1;
        // sketch_dimension = min(k + oversampling, min(nrows, ncols)) = min(1+1, min(2,1)) = min(2,1) = 1
        let sketch_oversampling_count = 1; 
        
        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), num_components_target_k, sketch_oversampling_count, 1, 42
        ).unwrap();
        // If sketch_dimension is 1, SVD of B might give 1 component.
        // The number of columns should be min(num_components_target_k, actual_computed_components)
        assert_eq!(result.shape(), &[1, 1]); // 1 sample, 1 component
    }

    #[test]
    fn test_rsvd_more_components_than_possible() {
        // Request more components than min(features, samples)
        let matrix_fs = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]); // 3 features, 2 samples
        let num_components_target_k = 5; // Request 5 PCs
        // sketch_dimension = min(k_target + oversampling, min(nrows,ncols))
        //                  = min(5 + 2, min(3,2)) = min(7, 2) = 2
        // So, at most 2 components can be computed from SVD of B.
        // Final result will be num_components_target_k.min(computed_from_svd_b) = 5.min(2) = 2
        let sketch_oversampling_count = 2;
        
        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), num_components_target_k, sketch_oversampling_count, 1, 42
        ).unwrap();
        assert_eq!(result.shape(), &[2, 2]); // num_samples x actual_components_computed
    }
    
    #[test]
    fn test_rsvd_tall_skinny_matrix() { // D >> N, common in genetics for A_eigen_std_star.t()
                                        // Here matrix_features_by_samples is A_eigen_std_star, so features >> samples
        let num_features = 100;
        let num_samples = 5;
        let matrix_fs = create_matrix(num_features, num_samples, 1.0);
        let num_components_target_k = 3;
        let sketch_oversampling_count = 2; // sketch_dim = min(3+2, min(100,5)) = min(5,5) = 5
        
        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), num_components_target_k, sketch_oversampling_count, 1, 42
        ).unwrap();
        assert_eq!(result.shape(), &[num_samples, num_components_target_k]);
    }

    #[test]
    fn test_rsvd_short_fat_matrix() { // D << N
        let num_features = 5;
        let num_samples = 100;
        let matrix_fs = create_matrix(num_features, num_samples, 1.0);
        let num_components_target_k = 3;
        let sketch_oversampling_count = 2; // sketch_dim = min(3+2, min(5,100)) = min(5,5) = 5

        let result = EigenSNPCoreAlgorithm::perform_randomized_svd_for_scores(
            &matrix_fs.view(), num_components_target_k, sketch_oversampling_count, 1, 42
        ).unwrap();
        assert_eq!(result.shape(), &[num_samples, num_components_target_k]);
    }
}

#[cfg(test)]
mod test_project_all_samples_onto_local_bases {
    use super::*; // Imports TestGenotypeData, EigenSNPCoreAlgorithmConfig, etc.
    use crate::eigensnp::{LdBlockSpecification, PcaSnpId, QcSampleId, EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, PerBlockLocalSnpBasis, LdBlockListId, RawCondensedFeatures};
    use ndarray::{arr2, array, Array2, Axis, s};
    use float_cmp::assert_approx_eq;

    // Helper to create TestGenotypeData
    fn create_test_genotype_data(data: Array2<f32>) -> TestGenotypeData {
        TestGenotypeData::new(data)
    }

    // Helper to create PerBlockLocalSnpBasis
    fn create_local_basis(block_id: usize, basis_vectors: Array2<f32>) -> PerBlockLocalSnpBasis {
        PerBlockLocalSnpBasis {
            block_list_id: LdBlockListId(block_id),
            basis_vectors,
        }
    }

    #[test]
    fn test_project_basic_case() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);

        // Genotype data: 3 SNPs, 4 Samples
        let genotype_data_arr = arr2(&[[1.0, 2.0, 3.0, 4.0],
                                       [5.0, 6.0, 7.0, 8.0],
                                       [9.0, 10.0, 11.0, 12.0]]);
        let genotype_accessor = create_test_genotype_data(genotype_data_arr.clone());
        let num_total_qc_samples = genotype_accessor.num_qc_samples();

        // LD Block Specs: One block with all 3 SNPs
        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: vec![PcaSnpId(0), PcaSnpId(1), PcaSnpId(2)],
            }
        ];

        // Local SNP Basis: 3 SNPs, 2 components for this block
        // V_block1 (SNPs x Components) = (3x2)
        // Basis vectors are columns. For projection, we need V_block1.t()
        let basis_vectors_b1 = arr2(&[[0.1, 0.4],
                                      [0.2, 0.5],
                                      [0.3, 0.6]]); 
        let all_local_bases = vec![
            create_local_basis(0, basis_vectors_b1.clone())
        ];

        let raw_condensed_features_result = algorithm.project_all_samples_onto_local_bases(
            &genotype_accessor,
            &ld_block_specs,
            &all_local_bases,
            num_total_qc_samples
        ).unwrap();

        // Expected output: (V_block1.t() * X_block1)
        // X_block1 is the genotype data for this block (3x4 SNPs x Samples)
        // basis_vectors_b1.t() is (2x3 Components x SNPs)
        // Expected result shape: (2 components_total, 4 samples)
        let expected_projection = basis_vectors_b1.t().dot(&genotype_data_arr);

        assert_eq!(raw_condensed_features_result.data.shape(), &[2, 4]);
        for i in 0..expected_projection.nrows() {
            for j in 0..expected_projection.ncols() {
                assert_approx_eq!(f32, raw_condensed_features_result.data[[i,j]], expected_projection[[i,j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_project_multiple_blocks() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);

        // Genotype data: 4 SNPs, 3 Samples
        // SNP0, SNP1 for Block1; SNP2, SNP3 for Block2
        let genotype_data_arr = arr2(&[[1., 2., 3.],  // SNP0
                                       [4., 5., 6.],  // SNP1
                                       [7., 8., 9.],  // SNP2
                                       [10.,11.,12.]]);// SNP3
        let genotype_accessor = create_test_genotype_data(genotype_data_arr.clone());
        let num_total_qc_samples = genotype_accessor.num_qc_samples();

        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: vec![PcaSnpId(0), PcaSnpId(1)], // SNPs 0, 1
            },
            LdBlockSpecification {
                user_defined_block_tag: "block2".to_string(),
                pca_snp_ids_in_block: vec![PcaSnpId(2), PcaSnpId(3)], // SNPs 2, 3
            }
        ];

        // Local bases
        // Block 1: 2 SNPs, 1 component. Basis V_b1: (2 SNPs x 1 Comp)
        let basis_b1 = arr2(&[[0.5], [0.5]]); 
        // Block 2: 2 SNPs, 2 components. Basis V_b2: (2 SNPs x 2 Comp)
        let basis_b2 = arr2(&[[0.7, 0.3], [0.3, 0.7]]); 
        
        let all_local_bases = vec![
            create_local_basis(0, basis_b1.clone()), // block_list_id must match index
            create_local_basis(1, basis_b2.clone())
        ];

        let raw_condensed_features_result = algorithm.project_all_samples_onto_local_bases(
            &genotype_accessor,
            &ld_block_specs,
            &all_local_bases,
            num_total_qc_samples
        ).unwrap();

        // Expected total components = 1 (from b1) + 2 (from b2) = 3
        assert_eq!(raw_condensed_features_result.data.shape(), &[3, 3]); // 3 total_condensed_features, 3 samples

        // Calculate expected projections manually
        // Block 1 projection: V_b1.t() * X_b1
        let x_b1 = genotype_data_arr.slice(s![0..2, ..]); // SNPs 0,1 (2x3)
        let expected_proj_b1 = basis_b1.t().dot(&x_b1); // (1x2) * (2x3) = (1x3)
        
        // Block 2 projection: V_b2.t() * X_b2
        let x_b2 = genotype_data_arr.slice(s![2..4, ..]); // SNPs 2,3 (2x3)
        let expected_proj_b2 = basis_b2.t().dot(&x_b2); // (2x2) * (2x3) = (2x3)

        // Check results
        // First row of condensed features should be from block1's projection
        for j in 0..num_total_qc_samples {
            assert_approx_eq!(f32, raw_condensed_features_result.data[[0, j]], expected_proj_b1[[0, j]], epsilon = 1e-6);
        }
        // Next two rows of condensed features should be from block2's projection
        for i in 0..expected_proj_b2.nrows() {
             for j in 0..num_total_qc_samples {
                assert_approx_eq!(f32, raw_condensed_features_result.data[[1 + i, j]], expected_proj_b2[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_project_empty_local_bases() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_accessor = create_test_genotype_data(arr2(&[[1.0, 2.0],[3.0, 4.0]]));
        let num_total_qc_samples = genotype_accessor.num_qc_samples();

        let ld_block_specs = vec![
            LdBlockSpecification { // A block, but its corresponding basis might be empty
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: vec![PcaSnpId(0), PcaSnpId(1)],
            }
        ];
        // All local bases result in zero components (e.g. basis_vectors.ncols() is 0)
        let all_local_bases = vec![
            create_local_basis(0, Array2::<f32>::zeros((2,0))) // 2 SNPs, 0 components
        ];

        let raw_condensed_features_result = algorithm.project_all_samples_onto_local_bases(
            &genotype_accessor,
            &ld_block_specs,
            &all_local_bases,
            num_total_qc_samples
        ).unwrap();
        
        // Expect 0 total condensed features
        assert_eq!(raw_condensed_features_result.data.nrows(), 0);
        assert_eq!(raw_condensed_features_result.data.ncols(), num_total_qc_samples);
    }

    #[test]
    fn test_project_block_with_no_snps_skipped() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
         let genotype_data_arr = arr2(&[[1., 2., 3.], [4., 5., 6.]]); // 2 SNPs, 3 Samples
        let genotype_accessor = create_test_genotype_data(genotype_data_arr.clone());
        let num_total_qc_samples = genotype_accessor.num_qc_samples();

        let ld_block_specs = vec![
            LdBlockSpecification { // Block 1: No SNPs
                user_defined_block_tag: "block_empty_snps".to_string(),
                pca_snp_ids_in_block: Vec::new(),
            },
            LdBlockSpecification { // Block 2: Has SNPs (0 and 1 from genotype_data_arr)
                user_defined_block_tag: "block_with_snps".to_string(),
                pca_snp_ids_in_block: vec![PcaSnpId(0), PcaSnpId(1)],
            }
        ];
        
        // Basis for block 2: 2 SNPs, 1 component. V_b2: (2 SNPs x 1 Comp)
        let basis_b2 = arr2(&[[0.5], [0.5]]); 
        let all_local_bases = vec![
            // Basis for empty block (block_list_id 0). This should have 0 SNPs, 0 components.
            create_local_basis(0, Array2::<f32>::zeros((0,0))), 
            create_local_basis(1, basis_b2.clone()) // Basis for block_with_snps (block_list_id 1)
        ];

        let raw_condensed_features_result = algorithm.project_all_samples_onto_local_bases(
            &genotype_accessor,
            &ld_block_specs,
            &all_local_bases,
            num_total_qc_samples
        ).unwrap();

        // Expected total components = 1 (from b2, b1 is skipped as it has 0 SNPs)
        assert_eq!(raw_condensed_features_result.data.shape(), &[1, 3]);

        // X_b2 is the full genotype_data_arr as PcaSnpId(0) and PcaSnpId(1) map to its rows.
        let x_b2 = genotype_data_arr.slice(s![0..2, ..]); // (2x3)
        let expected_proj_b2 = basis_b2.t().dot(&x_b2); // (1x2) * (2x3) = (1x3)

        for j in 0..num_total_qc_samples {
            assert_approx_eq!(f32, raw_condensed_features_result.data[[0, j]], expected_proj_b2[[0, j]], epsilon = 1e-6);
        }
    }
}

#[cfg(test)]
mod test_learn_all_ld_block_local_bases {
    use super::*; // Imports TestGenotypeData, EigenSNPCoreAlgorithmConfig, etc.
    use crate::eigensnp::{LdBlockSpecification, PcaSnpId, QcSampleId, EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, PerBlockLocalSnpBasis, LdBlockListId};
    use ndarray::{arr2, Array2};

    fn create_mock_genotype_data(snp_count: usize, sample_count: usize) -> TestGenotypeData {
        let mut genotypes = Array2::<f32>::zeros((snp_count, sample_count));
        // Fill with some non-trivial data to avoid all zeros in SVD if possible
        // Using a simple pattern that ensures variance for small matrices
        for i in 0..snp_count {
            for j in 0..sample_count {
                genotypes[[i, j]] = (i as f32 * 0.5 + j as f32 * 0.25).sin() + (j as f32 * 0.5).cos();
            }
        }
        // Ensure data is not all zeros and has some variability
        if snp_count > 0 && sample_count > 0 && genotypes.sum() == 0.0 {
             for i in 0..snp_count {
                for j in 0..sample_count {
                    genotypes[[i,j]] = (i+j) as f32;
                }
            }
        }
        if snp_count == 1 && sample_count > 1 { // Ensure variance for single SNP case
            for j in 0..sample_count { genotypes[[0,j]] = j as f32; }
        }
        if sample_count == 1 && snp_count > 1 { // Ensure variance for single sample case
             for i in 0..snp_count { genotypes[[i,0]] = i as f32; }
        }


        TestGenotypeData::new(genotypes)
    }

    #[test]
    fn test_single_ld_block_basic() {
        let config = EigenSNPCoreAlgorithmConfig {
            components_per_ld_block: 2,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        
        let genotype_data = create_mock_genotype_data(5, 10); // 5 SNPs, 10 samples
        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..5).map(PcaSnpId).collect(),
            }
        ];
        let subset_sample_ids: Vec<QcSampleId> = (0..10).map(QcSampleId).collect();

        let result = algorithm.learn_all_ld_block_local_bases(&genotype_data, &ld_block_specs, &subset_sample_ids).unwrap();
        
        assert_eq!(result.len(), 1);
        let basis = &result[0];
        assert_eq!(basis.block_list_id, LdBlockListId(0));
        // Expected components = min(config.components_per_ld_block, num_snps_in_block, num_samples)
        // num_snps_in_block = 5, num_samples = 10, config.components_per_ld_block = 2
        let expected_components = config.components_per_ld_block.min(5).min(10);
        assert_eq!(basis.basis_vectors.nrows(), 5); // num_snps_in_block
        assert_eq!(basis.basis_vectors.ncols(), expected_components); 
    }

    #[test]
    fn test_multiple_ld_blocks_one_empty() {
        let config = EigenSNPCoreAlgorithmConfig {
            components_per_ld_block: 3,
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        
        let genotype_data = create_mock_genotype_data(10, 20); // 10 SNPs, 20 samples
        let ld_block_specs = vec![
            LdBlockSpecification { // Block 1: SNPs 0-4
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..5).map(PcaSnpId).collect(),
            },
            LdBlockSpecification { // Block 2: No SNPs
                user_defined_block_tag: "block2_empty".to_string(),
                pca_snp_ids_in_block: Vec::new(),
            },
            LdBlockSpecification { // Block 3: SNPs 5-9
                user_defined_block_tag: "block3".to_string(),
                pca_snp_ids_in_block: (5..10).map(PcaSnpId).collect(),
            }
        ];
        let subset_sample_ids: Vec<QcSampleId> = (0..20).map(QcSampleId).collect();

        let result = algorithm.learn_all_ld_block_local_bases(&genotype_data, &ld_block_specs, &subset_sample_ids).unwrap();
        
        assert_eq!(result.len(), 3);

        // Block 1
        assert_eq!(result[0].block_list_id, LdBlockListId(0));
        assert_eq!(result[0].basis_vectors.nrows(), 5); // 5 SNPs in block1
        let expected_components_b1 = config.components_per_ld_block.min(5).min(20);
        assert_eq!(result[0].basis_vectors.ncols(), expected_components_b1);

        // Block 2 (empty)
        assert_eq!(result[1].block_list_id, LdBlockListId(1));
        assert_eq!(result[1].basis_vectors.nrows(), 0); // 0 SNPs
        assert_eq!(result[1].basis_vectors.ncols(), 0); // 0 components

        // Block 3
        assert_eq!(result[2].block_list_id, LdBlockListId(2));
        assert_eq!(result[2].basis_vectors.nrows(), 5); // 5 SNPs in block3
        let expected_components_b3 = config.components_per_ld_block.min(5).min(20);
        assert_eq!(result[2].basis_vectors.ncols(), expected_components_b3);
    }

    #[test]
    fn test_components_capped_by_snps_or_samples() {
        let config = EigenSNPCoreAlgorithmConfig {
            components_per_ld_block: 10, // Request more components than available
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config.clone());
        
        // Case 1: Capped by SNPs
        let genotype_data_snps_cap = create_mock_genotype_data(3, 20); // 3 SNPs, 20 samples
        let ld_block_specs_snps_cap = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block_snps_cap".to_string(),
                pca_snp_ids_in_block: (0..3).map(PcaSnpId).collect(),
            }
        ];
        let subset_sample_ids_snps_cap: Vec<QcSampleId> = (0..20).map(QcSampleId).collect();
        let result_snps_cap = algorithm.learn_all_ld_block_local_bases(&genotype_data_snps_cap, &ld_block_specs_snps_cap, &subset_sample_ids_snps_cap).unwrap();
        assert_eq!(result_snps_cap[0].basis_vectors.ncols(), 3); // Expected: min(10, 3, 20) = 3

        // Case 2: Capped by samples
        let genotype_data_samples_cap = create_mock_genotype_data(15, 4); // 15 SNPs, 4 samples
        let ld_block_specs_samples_cap = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block_samples_cap".to_string(),
                pca_snp_ids_in_block: (0..15).map(PcaSnpId).collect(),
            }
        ];
        let subset_sample_ids_samples_cap: Vec<QcSampleId> = (0..4).map(QcSampleId).collect();
        let result_samples_cap = algorithm.learn_all_ld_block_local_bases(&genotype_data_samples_cap, &ld_block_specs_samples_cap, &subset_sample_ids_samples_cap).unwrap();
        assert_eq!(result_samples_cap[0].basis_vectors.ncols(), 4); // Expected: min(10, 15, 4) = 4
    }

    #[test]
    fn test_empty_subset_sample_ids_with_snps_errors() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        
        let genotype_data = create_mock_genotype_data(5, 10);
        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..5).map(PcaSnpId).collect(), // Block has SNPs
            }
        ];
        let empty_subset_sample_ids: Vec<QcSampleId> = Vec::new();

        let result = algorithm.learn_all_ld_block_local_bases(&genotype_data, &ld_block_specs, &empty_subset_sample_ids);
        assert!(result.is_err()); 
        // Check for specific error message content
        let error_message = result.err().unwrap().to_string();
        assert!(error_message.contains("Subset sample IDs for local basis learning cannot be empty if LD blocks contain SNPs."));
    }
    
    #[test]
    fn test_no_snps_in_any_block() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_mock_genotype_data(0, 10); // No SNPs in overall data
        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block_no_snps".to_string(),
                pca_snp_ids_in_block: Vec::new(),
            }
        ];
        let subset_sample_ids: Vec<QcSampleId> = (0..10).map(QcSampleId).collect();

        let result = algorithm.learn_all_ld_block_local_bases(&genotype_data, &ld_block_specs, &subset_sample_ids).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].basis_vectors.nrows(), 0);
        assert_eq!(result[0].basis_vectors.ncols(), 0);
    }

    #[test]
    fn test_zero_components_requested() {
        let config = EigenSNPCoreAlgorithmConfig {
            components_per_ld_block: 0, // Request zero components
            ..Default::default()
        };
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_mock_genotype_data(5, 10);
        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block1".to_string(),
                pca_snp_ids_in_block: (0..5).map(PcaSnpId).collect(),
            }
        ];
        let subset_sample_ids: Vec<QcSampleId> = (0..10).map(QcSampleId).collect();
        let result = algorithm.learn_all_ld_block_local_bases(&genotype_data, &ld_block_specs, &subset_sample_ids).unwrap();
        assert_eq!(result[0].basis_vectors.ncols(), 0); // Expect 0 components
    }

    #[test]
    fn test_empty_subset_sample_ids_and_no_snps_in_block_is_ok() {
        let config = EigenSNPCoreAlgorithmConfig::default();
        let algorithm = EigenSNPCoreAlgorithm::new(config);
        let genotype_data = create_mock_genotype_data(0, 0); // No samples, no SNPs in data
        let ld_block_specs = vec![
            LdBlockSpecification {
                user_defined_block_tag: "block_empty_snps".to_string(),
                pca_snp_ids_in_block: Vec::new(), // Block has no SNPs
            }
        ];
        let empty_subset_sample_ids: Vec<QcSampleId> = Vec::new(); // No subset samples

        let result = algorithm.learn_all_ld_block_local_bases(&genotype_data, &ld_block_specs, &empty_subset_sample_ids).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].basis_vectors.nrows(), 0);
        assert_eq!(result[0].basis_vectors.ncols(), 0);
    }
}

impl PcaReadyGenotypeAccessor for TestGenotypeData {
    fn get_standardized_snp_sample_block(
        &self,
        snp_ids: &[PcaSnpId],
        sample_ids: &[QcSampleId],
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let mut result_matrix = Array2::zeros((snp_ids.len(), sample_ids.len()));
        for (i, pca_snp_id) in snp_ids.iter().enumerate() {
            let row_index = pca_snp_id.0; // Assuming PcaSnpId(idx) is the row index
            if row_index >= self.num_pca_snps() {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("PcaSnpId {:?} out of bounds for num_pca_snps {}", pca_snp_id, self.num_pca_snps()))));
            }
            for (j, qc_sample_id) in sample_ids.iter().enumerate() {
                let col_index = qc_sample_id.0; // Assuming QcSampleId(idx) is the col index
                if col_index >= self.num_qc_samples() {
                     return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("QcSampleId {:?} out of bounds for num_qc_samples {}", qc_sample_id, self.num_qc_samples()))));
                }
                result_matrix[[i, j]] = self.genotypes[[row_index, col_index]];
            }
        }
        Ok(result_matrix)
    }

    fn num_pca_snps(&self) -> usize {
        self.num_pca_snps
    }

    fn num_qc_samples(&self) -> usize {
        self.num_qc_samples
    }
}

#[cfg(test)]
mod test_standardize_raw_condensed_features {
    use super::*; // Imports TestGenotypeData, etc.
    use crate::eigensnp::RawCondensedFeatures; // Already in eigensnp_tests.rs use statements
    use crate::eigensnp::standardize_raw_condensed_features; // Now accessible due to pub(crate)
    use ndarray::{arr2, Array2, Axis};
    use float_cmp::assert_approx_eq; // For comparing floating point numbers

    // Helper function to check if a row is standardized
    fn assert_row_standardized(row: ndarray::ArrayView1<f32>, expected_mean: f32, expected_std_dev: f32) {
        let mean = row.mean().unwrap_or(0.0);
        assert_approx_eq!(f32, mean, expected_mean, epsilon = 1e-6);

        if row.len() > 1 {
            let std_dev = row.std(1.0); // ddof = 1 for sample standard deviation
             assert_approx_eq!(f32, std_dev, expected_std_dev, epsilon = 1e-6);
        } else if row.len() == 1 && expected_std_dev == 0.0 {
            // For a single element, std dev is 0.
            // The function might produce 0 or keep original if it was already 0.
            // The standardize_raw_condensed_features sets row to 0.0 if std_dev is near zero.
             assert_approx_eq!(f32, row[0], 0.0, epsilon = 1e-6);
        }
    }


    #[test]
    fn test_standardize_basic_matrix() {
        let raw_data = arr2(&[[1.0, 2.0, 3.0], 
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]]);
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        
        for row in standardized_result.data.axis_iter(Axis(0)) {
            assert_row_standardized(row, 0.0, 1.0);
        }
    }

    #[test]
    fn test_standardize_constant_value_row() {
        let raw_data = arr2(&[[1.0, 2.0, 3.0], 
                              [5.0, 5.0, 5.0], // Constant row
                              [7.0, 8.0, 9.0]]);
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        
        assert_row_standardized(standardized_result.data.row(0), 0.0, 1.0);
        // The constant row should become all zeros
        for &val in standardized_result.data.row(1) {
            assert_approx_eq!(f32, val, 0.0, epsilon = 1e-6);
        }
        assert_row_standardized(standardized_result.data.row(2), 0.0, 1.0);
    }

    #[test]
    fn test_standardize_single_sample_matrix() {
        // If num_samples <= 1, the function should fill with 0.0
        let raw_data = arr2(&[[10.0], 
                              [20.0], 
                              [30.0]]);
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        
        for row in standardized_result.data.axis_iter(Axis(0)) {
            for &val in row {
                 assert_approx_eq!(f32, val, 0.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_standardize_empty_matrix() {
        let raw_data = Array2::<f32>::zeros((0, 0));
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        assert_eq!(standardized_result.data.nrows(), 0);
        assert_eq!(standardized_result.data.ncols(), 0);

        let raw_data_rows_no_cols = Array2::<f32>::zeros((3, 0));
        let raw_features_rows_no_cols = RawCondensedFeatures { data: raw_data_rows_no_cols };
        let standardized_result_rows_no_cols = standardize_raw_condensed_features(raw_features_rows_no_cols).unwrap();
        assert_eq!(standardized_result_rows_no_cols.data.nrows(), 3);
        assert_eq!(standardized_result_rows_no_cols.data.ncols(), 0);
    }
    
    #[test]
    fn test_standardize_single_row_matrix() {
        let raw_data = arr2(&[[1.0, 2.0, 3.0]]);
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        assert_row_standardized(standardized_result.data.row(0), 0.0, 1.0);
    }

    #[test]
    fn test_standardize_single_column_matrix_multiple_rows() {
        // This is the same as test_standardize_single_sample_matrix due to num_samples <= 1 behavior
        let raw_data = arr2(&[[1.0], [2.0], [3.0]]);
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        for val in standardized_result.data.iter() {
            assert_approx_eq!(f32, *val, 0.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_standardize_row_near_zero_std_dev() {
        // After mean centering, if std dev is very small, it should be treated as zero.
        // E.g., mean = 2.0. Centered: [-1e-8, 0.0, 1e-8]. Std dev is tiny.
        let val = 1e-8;
        let raw_data = arr2(&[[2.0 - val, 2.0, 2.0 + val]]);
        let raw_features = RawCondensedFeatures { data: raw_data };
        let standardized_result = standardize_raw_condensed_features(raw_features).unwrap();
        // This row should become all zeros
        for &v in standardized_result.data.row(0) {
            assert_approx_eq!(f32, v, 0.0, epsilon = 1e-6);
        }
    }
}
