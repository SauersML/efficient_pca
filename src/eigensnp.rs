use ndarray::{s, Array1, Array2, Axis, ArrayView2};
use ndarray_linalg::{Eigh, UPLO, QR as NdarrayQR};
use rayon::prelude::*;
use std::error::Error;
use log::{info, debug, trace};

// --- Core Index Types for Enhanced Type Safety ---

/// Identifies a SNP included in the PCA (post-QC and part of an LD block).
/// This index is relative to the final list of SNPs used in the analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PcaSnpId(pub usize);

/// Identifies a sample included in the PCA (post-QC).
/// This index is relative to the final list of QC'd samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QcSampleId(pub usize);

/// Identifies an LD Block from the input list of `LdBlockSpecification`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LdBlockListId(pub usize);

/// Identifies a row (a condensed feature) in the condensed feature matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CondensedFeatureId(pub usize);

/// Identifies one of the K final computed Principal Components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrincipalComponentId(pub usize);

// --- Trait for Abstracting Genotype Data Access ---

/// Defines how the EigenSNP PCA algorithm accesses globally standardized genotype data.
/// The implementor of this trait is responsible for handling the actual data source
/// and applying pre-calculated global mean (mu_j) and
/// standard deviation (sigma_j) for each SNP j to provide X_ji = (G_qc,ji - mu_j) / sigma_j.
pub trait PcaReadyGenotypeAccessor: Sync {
    /// Fetches a block of globally standardized genotypes.
    /// The output matrix orientation is SNPs as rows, Samples as columns.
    ///
    /// # Arguments
    /// * `snp_ids`: A slice of `PcaSnpId` specifying which SNPs to fetch.
    /// * `sample_ids`: A slice of `QcSampleId` specifying which samples to fetch.
    ///
    /// # Returns
    /// An `Array2<f32>` of shape `(snp_ids.len(), sample_ids.len())`.
    fn get_standardized_snp_sample_block(
        &self,
        snp_ids: &[PcaSnpId],
        sample_ids: &[QcSampleId],
    ) -> Result<Array2<f32>, Box<dyn Error>>;

    /// Returns the total number of SNPs available for PCA (i.e., D_blocked,
    /// those SNPs that passed QC and are part of defined LD blocks).
    fn num_pca_snps(&self) -> usize;

    /// Returns the total number of samples available for PCA (i.e., N, after QC).
    fn num_qc_samples(&self) -> usize;
}

// --- Input Configuration Structures ---

/// Specification for a single Linkage Disequilibrium (LD) block.
#[derive(Clone, Debug)]
pub struct LdBlockSpecification {
    /// User-defined tag for identifying the block, primarily for tracking or logging.
    /// This field must be populated, for instance, with a block's genomic coordinates or a sequential ID.
    pub user_defined_block_tag: String,
    /// List of `PcaSnpId`s that belong to this LD block.
    pub pca_snp_ids_in_block: Vec<PcaSnpId>,
}

impl LdBlockSpecification {
    /// Returns the number of SNPs in this LD block.
    pub fn num_snps_in_block(&self) -> usize {
        self.pca_snp_ids_in_block.len()
    }
}

/// Configuration for the core EigenSNP PCA algorithm's internal parameters.
/// These parameters define the behavior of various algorithmic stages.
#[derive(Clone, Debug)]
pub struct EigenSNPCoreAlgorithmConfig {
    /// Factor of total samples (N) to consider for the N_s subset size when learning local bases.
    pub subset_factor_for_local_basis_learning: f64,
    /// Minimum number of samples for the N_s subset.
    pub min_subset_size_for_local_basis_learning: usize,
    /// Maximum number of samples for the N_s subset.
    pub max_subset_size_for_local_basis_learning: usize,

    /// Number of local eigenSNPs (principal components) to extract per LD block (c_p).
    pub components_per_ld_block: usize,

    /// Target number of global Principal Components (K) to compute.
    pub target_num_global_pcs: usize,
    /// Number of additional random dimensions for sketching in the global RSVD stage (L_glob = K_target + this).
    pub global_pca_sketch_oversampling: usize,
    /// Number of power iterations for the global RSVD on the condensed feature matrix.
    pub global_pca_num_power_iterations: usize,

    /// Seed for the random number generator used in RSVD stages.
    pub random_seed: u64,
}

impl Default for EigenSNPCoreAlgorithmConfig {
    /// Provides sensible default parameters for the EigenSNP PCA algorithm.
    fn default() -> Self {
        EigenSNPCoreAlgorithmConfig {
            subset_factor_for_local_basis_learning: 0.075,
            min_subset_size_for_local_basis_learning: 10_000,
            max_subset_size_for_local_basis_learning: 40_000,
            components_per_ld_block: 7,
            target_num_global_pcs: 15,
            global_pca_sketch_oversampling: 10,
            global_pca_num_power_iterations: 2,
            random_seed: 42,
        }
    }
}

// --- Typed Intermediate Data Product Structs ---

/// Represents the learned local eigenSNP basis vectors for a single LD block.
#[derive(Debug)]
pub struct PerBlockLocalSnpBasis {
    /// Identifier linking back to the `LdBlockSpecification` list, corresponding to its index.
    pub block_list_id: LdBlockListId,
    /// Matrix of basis vectors (local eigenSNP loadings).
    /// Shape: `(num_snps_in_block, num_local_components_for_block)`
    pub basis_vectors: Array2<f32>,
}

/// Represents the raw condensed feature matrix (A_eigen_star) before row-wise standardization.
/// Its features (rows) are the projections of all samples onto the local eigenSNP bases.
#[derive(Debug)]
pub struct RawCondensedFeatures {
    /// Data matrix.
    /// Shape: `(total_condensed_features, num_qc_samples)`
    pub data: Array2<f32>,
}

impl RawCondensedFeatures {
    pub fn num_total_condensed_features(&self) -> usize { self.data.nrows() }
    pub fn num_samples(&self) -> usize { self.data.ncols() }
}

/// Represents the condensed feature matrix (A_eigen_std_star) after its features (rows)
/// have been standardized (mean-centered and scaled to unit variance).
#[derive(Debug)]
pub struct StandardizedCondensedFeatures {
    /// Data matrix.
    /// Shape: `(total_condensed_features, num_qc_samples)`
    pub data: Array2<f32>,
}

impl StandardizedCondensedFeatures {
    pub fn num_total_condensed_features(&self) -> usize { self.data.nrows() }
    pub fn num_samples(&self) -> usize { self.data.ncols() }
}

/// Represents the initial Principal Component scores for all N samples,
/// derived from the PCA on the `StandardizedCondensedFeatures`.
#[derive(Debug)]
pub struct InitialSamplePcScores {
    /// Scores matrix (U_scores_star).
    /// Shape: `(num_qc_samples, num_global_pcs_computed)`
    pub scores: Array2<f32>,
}

impl InitialSamplePcScores {
    pub fn num_samples(&self) -> usize { self.scores.nrows() }
    pub fn num_pcs_computed(&self) -> usize { self.scores.ncols() }
}

// --- Final Output Structure ---

/// Encapsulates the final results of the EigenSNP PCA computation.
#[derive(Debug)]
pub struct EigenSNPCoreOutput {
    /// Final SNP Principal Component Loadings (V_final_star).
    /// Shape: `(num_pca_snps, num_principal_components_computed)`
    pub final_snp_principal_component_loadings: Array2<f32>,
    /// Final Principal Component Scores for the N reference individuals (S_final_star).
    /// Columns are orthogonal.
    /// Shape: `(num_qc_samples, num_principal_components_computed)`
    pub final_sample_principal_component_scores: Array2<f32>,
    /// Sample Eigenvalues for each computed Principal Component (lambda_k).
    /// Shape: `(num_principal_components_computed)`
    pub final_principal_component_eigenvalues: Array1<f64>,

    /// Number of QC'd individuals used in the PCA (N).
    pub num_qc_samples_used: usize,
    /// Number of QC'd SNPs (within defined LD blocks) used in the PCA (D_blocked).
    pub num_pca_snps_used: usize,
    /// Actual number of Principal Components computed (K_computed <= K_target).
    pub num_principal_components_computed: usize,
}

// --- Main Algorithm Orchestrator Struct Definition ---

/// Orchestrates the EigenSNP PCA algorithm.
/// Holds the configuration and provides the main execution method.
#[derive(Debug, Clone)]
pub struct EigenSNPCoreAlgorithm {
    config: EigenSNPCoreAlgorithmConfig,
}

impl EigenSNPCoreAlgorithm {
    /// Creates a new `EigenSNPCoreAlgorithm` runner with the given configuration.
    pub fn new(config: EigenSNPCoreAlgorithmConfig) -> Self {
        Self { config }
    }
}









impl EigenSNPCoreAlgorithm {
    /// Learns local eigenSNP bases for each LD block using a subset of samples.
    fn learn_all_ld_block_local_bases<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specs: &[LdBlockSpecification],
        subset_sample_ids: &[QcSampleId],
    ) -> Result<Vec<PerBlockLocalSnpBasis>, Box<dyn Error>> {
        info!(
            "Learning local eigenSNP bases for {} LD blocks using N_s = {} samples.",
            ld_block_specs.len(),
            subset_sample_ids.len()
        );

        if subset_sample_ids.is_empty() {
            let any_snps_in_blocks = ld_block_specs.iter().any(|b| b.num_snps_in_block() > 0);
            if any_snps_in_blocks {
                return Err("Subset sample IDs cannot be empty if there are LD blocks with SNPs.".into());
            }
        }

        let local_bases_results: Vec<Result<PerBlockLocalSnpBasis, Box<dyn Error>>> =
            ld_block_specs
                .par_iter()
                .enumerate()
                .map(|(block_idx_val, block_spec)| {
                    let block_list_id = LdBlockListId(block_idx_val);
                    let num_snps_in_this_block = block_spec.num_snps_in_block();

                    if num_snps_in_this_block == 0 {
                        trace!("Block {:?} is empty of SNPs, creating empty basis.", block_list_id);
                        return Ok(PerBlockLocalSnpBasis {
                            block_list_id,
                            basis_vectors: Array2::<f32>::zeros((0, 0)),
                        });
                    }

                    // Fetch standardized genotype data for this block and the N_s subset
                    // x_subset_block_snps_x_samples has shape (num_snps_in_this_block, subset_sample_ids.len())
                    let x_subset_block_snps_x_samples =
                        genotype_data.get_standardized_snp_sample_block(
                            &block_spec.pca_snp_ids_in_block,
                            subset_sample_ids,
                        )?;
                    
                    let rank_limit_from_samples = if !subset_sample_ids.is_empty() { subset_sample_ids.len() } else { 0 };
                    let num_components_to_extract = self.config.components_per_ld_block
                        .min(num_snps_in_this_block)
                        .min(rank_limit_from_samples);

                    if num_components_to_extract == 0 {
                        debug!(
                            "Block {:?}: num components to extract is 0 (M_p={}, N_s={}, configured_cp={}), creating empty basis.", 
                            block_list_id, 
                            num_snps_in_this_block, 
                            subset_sample_ids.len(),
                            self.config.components_per_ld_block
                        );
                        return Ok(PerBlockLocalSnpBasis {
                            block_list_id,
                            basis_vectors: Array2::<f32>::zeros((num_snps_in_this_block, 0)),
                        });
                    }
                    
                    // Perform Direct SVD via X_s_p * X_s_p.T (M_p x M_p covariance matrix)
                    // Using f64 for covariance matrix and its eigendecomposition for better precision.
                    // x_subset_block_snps_x_samples is (M_p x N_s)
                    // x_subset_block_snps_x_samples.t() is (N_s x M_p)
                    let cov_matrix_snps_f64 = x_subset_block_snps_x_samples
                        .dot(&x_subset_block_snps_x_samples.t().mapv(|x_val| x_val as f64)); // M_p x M_p

                    let (_eigenvalues_f64, eigenvectors_f64_cols) =
                        cov_matrix_snps_f64.eigh(UPLO::Upper)
                        .map_err(|e| format!("Eigendecomposition failed for block {:?}: {}", block_list_id, e))?;
                    
                    // Eigenvalues from .eigh are ascending. Select the last `num_components_to_extract` eigenvectors.
                    let selected_eigenvectors_f64 = eigenvectors_f64_cols.slice_axis(
                        Axis(1),
                        ndarray::Slice::from((num_snps_in_this_block - num_components_to_extract)..num_snps_in_this_block),
                    );

                    // Convert the selected basis vectors to f32 for storage
                    let local_basis_vectors_f32 = selected_eigenvectors_f64.mapv(|x_val| x_val as f32);

                    trace!("Block {:?}: extracted {} local components.", block_list_id, num_components_to_extract);
                    Ok(PerBlockLocalSnpBasis {
                        block_list_id,
                        basis_vectors: local_basis_vectors_f32.into_owned(), // into_owned if slice_axis returns a view
                    })
                })
                .collect();

        // Collect results, propagating the first error if any occurred.
        let mut all_local_bases_final = Vec::with_capacity(ld_block_specs.len());
        for result_item in local_bases_results {
            all_local_bases_final.push(result_item?);
        }

        info!("Successfully learned local eigenSNP bases for all blocks.");
        Ok(all_local_bases_final)
    }

    /// Constructs the raw condensed feature matrix (A_eigen_star)
    /// by projecting all N samples onto the learned local eigenSNP bases.
    fn project_all_samples_onto_local_bases<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specs: &[LdBlockSpecification],
        all_local_bases: &[PerBlockLocalSnpBasis], 
        num_total_qc_samples: usize,
    ) -> Result<RawCondensedFeatures, Box<dyn Error>> {
        info!(
            "Projecting N={} samples onto local bases to construct condensed feature matrix.",
            num_total_qc_samples
        );

        let total_condensed_features: usize = all_local_bases.iter().map(|basis| basis.basis_vectors.ncols()).sum();

        if total_condensed_features == 0 {
            info!("Total condensed features M' is 0. Returning empty RawCondensedFeatures.");
            return Ok(RawCondensedFeatures {
                data: Array2::<f32>::zeros((0, num_total_qc_samples)),
            });
        }
        debug!("Total condensed features M' = {}", total_condensed_features);

        let mut raw_condensed_data_matrix =
            Array2::<f32>::zeros((total_condensed_features, num_total_qc_samples));
        let mut current_condensed_feature_row_offset = 0;

        let all_qc_sample_ids: Vec<QcSampleId> = (0..num_total_qc_samples).map(QcSampleId).collect();

        for block_list_idx_val in 0..ld_block_specs.len() {
            let block_spec = &ld_block_specs[block_list_idx_val];
            // Find the corresponding local_basis; assumes all_local_bases is in the same order
            // or that PerBlockLocalSnpBasis.block_list_id can be used for robust lookup if needed.
            // For this implementation, direct indexing is used assuming order.
            let local_basis_data = &all_local_bases[block_list_idx_val];
            
            let u_p_star_cp = &local_basis_data.basis_vectors; // Shape: (num_snps_in_block, num_local_components_this_block)
            let num_local_components_this_block = u_p_star_cp.ncols();

            if block_spec.num_snps_in_block() == 0 || num_local_components_this_block == 0 {
                trace!("Skipping block LdBlockListId({}) for projection: M_p={} or c_p=0.", block_list_idx_val, block_spec.num_snps_in_block());
                continue;
            }
            
            // Fetch X_p: standardized genotype data for this block for ALL N samples
            // Shape: (num_snps_in_block, num_total_qc_samples)
            let x_block_all_samples = genotype_data.get_standardized_snp_sample_block(
                &block_spec.pca_snp_ids_in_block,
                &all_qc_sample_ids,
            )?;

            // Compute S_p_star = U_p_star_cp.T @ X_block_all_samples
            // Shapes: (num_local_components_this_block, num_snps_in_block) @ (num_snps_in_block, num_total_qc_samples)
            // Result S_p_star shape: (num_local_components_this_block, num_total_qc_samples)
            let s_p_star = u_p_star_cp.t().dot(&x_block_all_samples);

            raw_condensed_data_matrix
                .slice_mut(s![
                    current_condensed_feature_row_offset
                        ..current_condensed_feature_row_offset + num_local_components_this_block,
                    ..
                ])
                .assign(&s_p_star);

            current_condensed_feature_row_offset += num_local_components_this_block;
        }
        
        info!("Constructed raw condensed feature matrix. Shape: {:?}", raw_condensed_data_matrix.dim());
        Ok(RawCondensedFeatures { data: raw_condensed_data_matrix })
    }

    /// Standardizes the features (rows) of the raw condensed feature matrix.
    fn standardize_rows_of_condensed_matrix(
        raw_features_input: RawCondensedFeatures, // Takes ownership
    ) -> Result<StandardizedCondensedFeatures, Box<dyn Error>> {
        let mut condensed_data = raw_features_input.data; // Moves data
        let num_features = condensed_data.nrows();
        let num_samples = condensed_data.ncols();

        info!(
            "Standardizing rows of condensed feature matrix ({} features, {} samples).",
            num_features, num_samples
        );

        if num_samples <= 1 { 
            if num_features > 0 && num_samples == 1 { 
                 condensed_data.fill(0.0f32);
            } 
            debug!("Number of samples ({}) is <= 1, standardization may result in zeros or is skipped if empty.", num_samples);
            return Ok(StandardizedCondensedFeatures { data: condensed_data });
        }

        condensed_data
            .axis_iter_mut(Axis(0)) // Iterate over rows (features)
            .par_bridge() 
            .for_each(|mut feature_row_view| {
                let mut sum_f64: f64 = 0.0;
                for val_ref in feature_row_view.iter() {
                    sum_f64 += *val_ref as f64;
                }
                let mean_f64 = sum_f64 / (num_samples as f64);
                let mean_f32 = mean_f64 as f32;

                feature_row_view.mapv_inplace(|x_val| x_val - mean_f32);

                let mut sum_sq_dev_f64: f64 = 0.0;
                for val_centered_ref in feature_row_view.iter() {
                    sum_sq_dev_f64 += (*val_centered_ref as f64).powi(2);
                }
                
                let variance_f64 = sum_sq_dev_f64 / (num_samples as f64 - 1.0); 
                let std_dev_f64 = variance_f64.sqrt();
                let std_dev_f32 = std_dev_f64 as f32;

                if std_dev_f32.abs() > 1e-7 { 
                    feature_row_view.mapv_inplace(|x_val| x_val / std_dev_f32);
                } else {
                    feature_row_view.fill(0.0f32);
                }
            });
        
        info!("Finished standardizing rows of condensed feature matrix.");
        Ok(StandardizedCondensedFeatures { data: condensed_data })
    }
}
