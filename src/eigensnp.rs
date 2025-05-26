use ndarray::{s, Array1, Array2, Axis, ArrayView2};
// Eigh, QR, SVDInto are replaced by backend calls. UPLO is handled by eigh_upper.
// use ndarray_linalg::{Eigh, UPLO, QR, SVDInto}; 
use crate::linalg_backends::{BackendEigh, BackendQR, BackendSVD, LinAlgBackendProvider};
// use crate::ndarray_backend::NdarrayLinAlgBackend; // Replaced by LinAlgBackendProvider
// use crate::linalg_backend_dispatch::LinAlgBackendProvider; // Now part of linalg_backends
use rayon::prelude::*;
use std::error::Error;
use log::{info, debug, trace, warn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// A thread-safe wrapper for standard dynamic errors,
/// so they implement `Send` and `Sync`.
type ThreadSafeStdError = Box<dyn Error + Send + Sync + 'static>;

// --- Core Index Types ---

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
    ) -> Result<Array2<f32>, ThreadSafeStdError>;

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


// --- Utility Functions ---

/// Standardizes each row (feature) of the input condensed feature matrix to have zero mean and unit variance.
/// Features with a standard deviation effectively zero (absolute value < 1e-7) after mean centering
/// will be filled with zeros. This is a common approach to handle constant features in PCA
/// to prevent division by zero and for numerical stability.
fn standardize_raw_condensed_features(
    raw_features_input: RawCondensedFeatures,
) -> Result<StandardizedCondensedFeatures, ThreadSafeStdError> {
    let mut condensed_data_matrix = raw_features_input.data;
    let num_total_condensed_features = condensed_data_matrix.nrows();
    let num_samples = condensed_data_matrix.ncols();

    info!(
        "Standardizing rows of condensed feature matrix ({} features, {} samples).",
        num_total_condensed_features, num_samples
    );

    if num_samples <= 1 {
        if num_total_condensed_features > 0 && num_samples == 1 {
            // If there's only one sample, variance is undefined (or zero).
            // Standardizing would lead to NaNs or division by zero.
            // Filling with 0.0 is a consistent way to handle this.
            condensed_data_matrix.fill(0.0f32);
        }
        debug!("Number of samples ({}) is <= 1 for condensed matrix; standardization results in zeros or is skipped if already empty.", num_samples);
        return Ok(StandardizedCondensedFeatures { data: condensed_data_matrix });
    }

    // Parallelize row-wise standardization
    condensed_data_matrix
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut feature_row| {
            // Calculate mean
            let mut sum_for_mean_f64: f64 = 0.0;
            for val_ref in feature_row.iter() {
                sum_for_mean_f64 += *val_ref as f64;
            }
            let mean_val_f64 = sum_for_mean_f64 / (num_samples as f64);
            let mean_val_f32 = mean_val_f64 as f32;

            // Mean center the row
            feature_row.mapv_inplace(|x_val| x_val - mean_val_f32);

            // Calculate standard deviation of the mean-centered row
            let mut sum_sq_deviations_f64: f64 = 0.0;
            for val_centered_ref in feature_row.iter() {
                sum_sq_deviations_f64 += (*val_centered_ref as f64).powi(2);
            }
            
            // Use (N-1) for sample variance calculation
            let variance_f64 = sum_sq_deviations_f64 / (num_samples as f64 - 1.0);
            let std_dev_f64 = variance_f64.sqrt();
            let std_dev_f32 = std_dev_f64 as f32;

            // Scale by standard deviation
            if std_dev_f32.abs() > 1e-7 { // Check against a small epsilon to avoid division by zero
                feature_row.mapv_inplace(|x_val| x_val / std_dev_f32);
            } else {
                // If standard deviation is effectively zero, the feature is constant.
                // Set all values in this row to 0.0.
                feature_row.fill(0.0f32);
            }
        });
    
    info!("Finished standardizing rows of condensed feature matrix.");
    Ok(StandardizedCondensedFeatures { data: condensed_data_matrix })
}


// --- Main Algorithm Orchestrator Struct Definition ---

/// Orchestrates the EigenSNP PCA algorithm.
/// Holds the configuration and provides the main execution method.
#[derive(Debug, Clone)]
pub struct EigenSNPCoreAlgorithm {
    config: EigenSNPCoreAlgorithmConfig,
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
    
    /// Number of additional random dimensions for sketching in the local RSVD stage (L_local = c_p + this).
    pub local_rsvd_sketch_oversampling: usize, 
    /// Number of power iterations for the local RSVD stage.
    pub local_rsvd_num_power_iterations: usize,

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
            local_rsvd_sketch_oversampling: 10, 
            local_rsvd_num_power_iterations: 1, 
            random_seed: 2025,
        }
    }
}

impl EigenSNPCoreAlgorithm {
    /// Creates a new `EigenSNPCoreAlgorithm` runner with the given configuration.
    pub fn new(config: EigenSNPCoreAlgorithmConfig) -> Self {
        Self { config }
    }

    // --- Main Public Execution Method ---

    /// Orchestrates the entire EigenSNP PCA workflow.
    pub fn compute_pca<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specifications: &[LdBlockSpecification],
    ) -> Result<EigenSNPCoreOutput, ThreadSafeStdError> {
        let num_total_qc_samples = genotype_data.num_qc_samples();
        let num_total_pca_snps = genotype_data.num_pca_snps();

        // Determine subset sample IDs based on config
        let desired_subset_sample_count = (self.config.subset_factor_for_local_basis_learning * num_total_qc_samples as f64).round() as usize;
        let clamped_min_subset_sample_count = desired_subset_sample_count.max(self.config.min_subset_size_for_local_basis_learning);
        let actual_subset_sample_count = clamped_min_subset_sample_count.min(self.config.max_subset_size_for_local_basis_learning).min(num_total_qc_samples);

        info!(
            "Starting EigenSNP PCA. Target PCs={}, Total Samples={}, Subset Samples (N_s)={}, Num LD Blocks={}",
            self.config.target_num_global_pcs,
            num_total_qc_samples,
            actual_subset_sample_count,
            ld_block_specifications.len()
        );
        let overall_start_time = std::time::Instant::now();

        // Input Validations
        if self.config.target_num_global_pcs == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Target number of global PCs must be greater than 0.").into());
        }
        if num_total_pca_snps > 0 && ld_block_specifications.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "LD block specifications cannot be empty if PCA SNPs are present.").into());
        }
        if num_total_qc_samples == 0 {
            warn!("Genotype data has zero QC samples. Returning empty PCA output.");
            return Ok(EigenSNPCoreOutput {
                final_snp_principal_component_loadings: Array2::zeros((num_total_pca_snps, 0)),
                final_sample_principal_component_scores: Array2::zeros((0, 0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: 0,
                num_pca_snps_used: num_total_pca_snps,
                num_principal_components_computed: 0,
            });
        }
        if num_total_pca_snps == 0 {
            warn!("Genotype data has zero PCA SNPs. Returning empty PCA output.");
            return Ok(EigenSNPCoreOutput {
                final_snp_principal_component_loadings: Array2::zeros((0, 0)),
                final_sample_principal_component_scores: Array2::zeros((num_total_qc_samples, 0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: num_total_qc_samples,
                num_pca_snps_used: 0,
                num_principal_components_computed: 0,
            });
        }

        let subset_sample_ids_selected: Vec<QcSampleId> = if actual_subset_sample_count > 0 {
            let mut rng_subset_selection = ChaCha8Rng::seed_from_u64(self.config.random_seed);
            let subset_indices: Vec<usize> = rand::seq::index::sample(&mut rng_subset_selection, num_total_qc_samples, actual_subset_sample_count).into_vec();
            subset_indices.into_iter().map(QcSampleId).collect()
        } else {
             if num_total_qc_samples > 0 && ld_block_specifications.iter().any(|b| b.num_snps_in_block() > 0) {
                 warn!("Calculated N_s is 0, but total samples > 0 and blocks have SNPs. This situation is problematic for learning local bases.");
                 return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Subset size (N_s) for local basis learning is 0, but samples and SNP blocks are present.").into());
             }
             Vec::new()
        };


        let local_bases_learning_start_time = std::time::Instant::now();
        let all_block_local_bases = self.learn_all_ld_block_local_bases(
            genotype_data,
            ld_block_specifications,
            &subset_sample_ids_selected,
        )?;
        info!("Learned local SNP bases in {:?}", local_bases_learning_start_time.elapsed());

        let condensed_matrix_construction_start_time = std::time::Instant::now();
        let raw_condensed_feature_matrix = self.project_all_samples_onto_local_bases(
            genotype_data,
            ld_block_specifications,
            &all_block_local_bases,
            num_total_qc_samples,
        )?;
        info!("Constructed raw condensed feature matrix in {:?}", condensed_matrix_construction_start_time.elapsed());

        let condensed_matrix_standardization_start_time = std::time::Instant::now();
        let standardized_condensed_feature_matrix =
            standardize_raw_condensed_features(raw_condensed_feature_matrix)?; // Updated call site
        info!("Standardized condensed feature matrix in {:?}", condensed_matrix_standardization_start_time.elapsed());

        let initial_global_pca_start_time = std::time::Instant::now();
        let initial_sample_pc_scores = self.compute_pca_on_standardized_condensed_features_via_rsvd(
            &standardized_condensed_feature_matrix,
        )?;
        info!("Computed initial global PCA on condensed features in {:?}", initial_global_pca_start_time.elapsed());

        let num_computed_initial_pcs = initial_sample_pc_scores.scores.ncols();
        if num_computed_initial_pcs == 0 {
            warn!("Initial PCA on condensed features yielded 0 components. Returning empty PCA output.");
            return Ok(EigenSNPCoreOutput {
                final_snp_principal_component_loadings: Array2::zeros((num_total_pca_snps,0)),
                final_sample_principal_component_scores: Array2::zeros((num_total_qc_samples,0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: num_total_qc_samples,
                num_pca_snps_used: num_total_pca_snps,
                num_principal_components_computed: 0,
            });
        }

        debug!("Starting Score-Guided Refinement with {} initial PCs.", num_computed_initial_pcs);
        let loadings_refinement_start_time = std::time::Instant::now();
        let final_snp_loadings = self.compute_refined_snp_loadings(
            genotype_data,
            &initial_sample_pc_scores,
        )?;
        info!("Computed refined SNP loadings in {:?}", loadings_refinement_start_time.elapsed());
        
        if final_snp_loadings.ncols() == 0 {
            warn!("Refined SNP loadings resulted in 0 components. Returning empty PCA output.");
            return Ok(EigenSNPCoreOutput {
                final_snp_principal_component_loadings: final_snp_loadings, 
                final_sample_principal_component_scores: Array2::zeros((num_total_qc_samples,0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: num_total_qc_samples,
                num_pca_snps_used: num_total_pca_snps,
                num_principal_components_computed: 0,
            });
        }

        let final_scores_eigenvalues_start_time = std::time::Instant::now();
        let (final_sample_scores, final_eigenvalues) =
            self.compute_final_scores_and_eigenvalues(
                genotype_data,
                &final_snp_loadings.view(),
                num_total_qc_samples,
            )?;
        info!("Computed final scores and eigenvalues in {:?}", final_scores_eigenvalues_start_time.elapsed());

        let num_principal_components_computed_final = final_snp_loadings.ncols();

        info!(
            "EigenSNP PCA completed in {:?}. Computed {} Principal Components.",
            overall_start_time.elapsed(),
            num_principal_components_computed_final
        );

        Ok(EigenSNPCoreOutput {
            final_snp_principal_component_loadings: final_snp_loadings,
            final_sample_principal_component_scores: final_sample_scores,
            final_principal_component_eigenvalues: final_eigenvalues,
            num_qc_samples_used: num_total_qc_samples,
            num_pca_snps_used: num_total_pca_snps, // This should be D_blocked, need to track it
            num_principal_components_computed: num_principal_components_computed_final,
        })
    }

    fn learn_all_ld_block_local_bases<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specs: &[LdBlockSpecification],
        subset_sample_ids: &[QcSampleId],
    ) -> Result<Vec<PerBlockLocalSnpBasis>, ThreadSafeStdError> {
        info!(
            "Learning local eigenSNP bases for {} LD blocks using N_subset = {} samples.",
            ld_block_specs.len(),
            subset_sample_ids.len()
        );

        if subset_sample_ids.is_empty() {
            let any_snps_in_blocks = ld_block_specs.iter().any(|b| b.num_snps_in_block() > 0);
            if any_snps_in_blocks {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Subset sample IDs for local basis learning cannot be empty if LD blocks contain SNPs.").into());
            }
        }

        let local_bases_results: Vec<Result<PerBlockLocalSnpBasis, ThreadSafeStdError>> =
            ld_block_specs
                .par_iter()
                .enumerate()
                .map(|(block_idx_val, block_spec)| -> Result<PerBlockLocalSnpBasis, ThreadSafeStdError> {
                    let block_list_id = LdBlockListId(block_idx_val);
                    let num_snps_in_this_block_spec = block_spec.num_snps_in_block();

                    if num_snps_in_this_block_spec == 0 {
                        trace!("Block ID {:?} ({}) is empty of SNPs, creating empty basis.", block_list_id, block_spec.user_defined_block_tag);
                        return Ok(PerBlockLocalSnpBasis {
                            block_list_id,
                            basis_vectors: Array2::<f32>::zeros((0, 0)),
                        });
                    }

                    let genotype_block_for_subset_samples =
                        genotype_data.get_standardized_snp_sample_block(
                            &block_spec.pca_snp_ids_in_block,
                            subset_sample_ids,
                        ).map_err(|e_accessor| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get standardized SNP/sample block for block ID {:?} ({}): {}", block_list_id, block_spec.user_defined_block_tag, e_accessor))) as ThreadSafeStdError)?;
                    let actual_num_snps_in_block = genotype_block_for_subset_samples.nrows();
                    let actual_num_subset_samples = genotype_block_for_subset_samples.ncols();
                    
                    let num_components_to_extract = self.config.components_per_ld_block
                        .min(actual_num_snps_in_block)
                        .min(if actual_num_subset_samples > 0 { actual_num_subset_samples } else { 0 });

                    if num_components_to_extract == 0 {
                        debug!(
                            "Block ID {:?} ({}): num components to extract is 0 (SNPs_in_block={}, N_subset={}, Configured_cp={}), creating empty basis.", 
                            block_list_id, 
                            block_spec.user_defined_block_tag,
                            actual_num_snps_in_block, 
                            actual_num_subset_samples,
                            self.config.components_per_ld_block
                        );
                        return Ok(PerBlockLocalSnpBasis {
                            block_list_id,
                            basis_vectors: Array2::<f32>::zeros((actual_num_snps_in_block, 0)),
                        });
                    }
                    
                    // Generate a local seed for RSVD, ensuring it varies per block
                    let local_seed = self.config.random_seed.wrapping_add(block_idx_val as u64);

                    // genotype_block_for_subset_samples is SNPs x Samples (M_p x N_s)
                    // We want U from SVD(A), which is M_p x c_p (SNP loadings for the block)
                    let local_basis_vectors_f32 = Self::perform_randomized_svd_for_loadings(
                        &genotype_block_for_subset_samples.view(),
                        num_components_to_extract,
                        self.config.local_rsvd_sketch_oversampling,
                        self.config.local_rsvd_num_power_iterations,
                        local_seed,
                    ).map_err(|e_rsvd| -> ThreadSafeStdError {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "Local RSVD failed for block ID {:?} ({}): {}",
                                block_list_id, block_spec.user_defined_block_tag, e_rsvd
                            ),
                        ))
                    })?;

                    trace!("Block ID {:?} ({}): extracted {} local components via rSVD.", block_list_id, block_spec.user_defined_block_tag, num_components_to_extract);
                    Ok(PerBlockLocalSnpBasis {
                        block_list_id,
                        basis_vectors: local_basis_vectors_f32, // Already owned Array2<f32>
                    })
                })
                .collect();

        let mut all_local_bases_collection = Vec::with_capacity(ld_block_specs.len());
        for result_item in local_bases_results {
            all_local_bases_collection.push(result_item?);
        }

        info!("Successfully learned local eigenSNP bases for all blocks.");
        Ok(all_local_bases_collection)
    }

    fn project_all_samples_onto_local_bases<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specs: &[LdBlockSpecification],
        all_local_bases: &[PerBlockLocalSnpBasis], 
        num_total_qc_samples: usize,
    ) -> Result<RawCondensedFeatures, ThreadSafeStdError> {
        info!(
            "Projecting {} total QC samples onto local bases to construct condensed feature matrix.",
            num_total_qc_samples
        );

        let total_num_condensed_features: usize = all_local_bases.iter().map(|basis| basis.basis_vectors.ncols()).sum();

        if total_num_condensed_features == 0 {
            info!("Total condensed features is 0. Returning empty RawCondensedFeatures.");
            return Ok(RawCondensedFeatures {
                data: Array2::<f32>::zeros((0, num_total_qc_samples)),
            });
        }
        debug!("Total number of condensed features (rows in A_eigen) = {}", total_num_condensed_features);

        let mut raw_condensed_data_matrix =
            Array2::<f32>::zeros((total_num_condensed_features, num_total_qc_samples));
        let mut current_condensed_feature_row_offset = 0;

        let all_qc_sample_ids: Vec<QcSampleId> = (0..num_total_qc_samples).map(QcSampleId).collect();

        for block_idx in 0..ld_block_specs.len() {
            let block_spec = &ld_block_specs[block_idx];
            let local_basis_data = &all_local_bases[block_idx]; // all_local_bases have to be ordered same as ld_block_specs
            
            let local_snp_basis_vectors = &local_basis_data.basis_vectors; 
            let num_components_this_block = local_snp_basis_vectors.ncols();

            if block_spec.num_snps_in_block() == 0 || num_components_this_block == 0 {
                trace!("Skipping block with tag '{}' for projection: num_snps={} or num_local_components=0.", block_spec.user_defined_block_tag, block_spec.num_snps_in_block());
                continue;
            }
            
            let genotype_data_for_block_all_samples = genotype_data.get_standardized_snp_sample_block(
                &block_spec.pca_snp_ids_in_block,
                &all_qc_sample_ids,
            ).map_err(|e_accessor| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get standardized SNP/sample block during projection for block '{}': {}", block_spec.user_defined_block_tag, e_accessor))) as ThreadSafeStdError)?;
            
            let projected_scores_for_block = local_snp_basis_vectors.t().dot(&genotype_data_for_block_all_samples);

            raw_condensed_data_matrix
                .slice_mut(s![
                    current_condensed_feature_row_offset
                        ..current_condensed_feature_row_offset + num_components_this_block,
                    ..
                ])
                .assign(&projected_scores_for_block);

            current_condensed_feature_row_offset += num_components_this_block;
        }
        
        info!("Constructed raw condensed feature matrix. Shape: {:?}", raw_condensed_data_matrix.dim());
        Ok(RawCondensedFeatures { data: raw_condensed_data_matrix })
    }

    fn compute_pca_on_standardized_condensed_features_via_rsvd(
        &self,
        standardized_condensed_features: &StandardizedCondensedFeatures,
    ) -> Result<InitialSamplePcScores, ThreadSafeStdError> {
        // Note: The original `perform_randomized_svd_for_scores` returned V (scores directly related to V_A).
        // Here we are consistent with that, requesting V (right singular vectors of A).
        let sample_scores_n_by_k = Self::perform_randomized_svd_for_scores(
            &standardized_condensed_features.data.view(), // A is M x N (features x samples)
            self.config.target_num_global_pcs,
            self.config.global_pca_sketch_oversampling,
            self.config.global_pca_num_power_iterations,
            self.config.random_seed,
        )?;
        Ok(InitialSamplePcScores { scores: sample_scores_n_by_k })
    }

    /// Computes the right singular vectors (V_A_approx, sample scores) of a matrix A using rSVD.
    /// A is M features x N samples. Output is N x K_eff.
    pub fn perform_randomized_svd_for_scores(
        matrix_features_by_samples: &ArrayView2<f32>, 
        num_components_target_k: usize,
        sketch_oversampling_count: usize,
        num_power_iterations: usize,
        random_seed: u64,
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let (_u_opt, _s_opt, v_opt) = Self::_internal_perform_rsvd(
            matrix_features_by_samples,
            num_components_target_k,
            sketch_oversampling_count,
            num_power_iterations,
            random_seed,
            false, // request_u_components
            false, // request_s_components
            true,  // request_v_components
        )?;

        v_opt.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Scores (V matrix) not computed or available from _internal_perform_rsvd",
            )) as ThreadSafeStdError
        })
    }

    /// Computes the left singular vectors (U_A_approx, feature loadings) of a matrix A using rSVD.
    /// A is M features x N samples. Output is M x K_eff.
    pub fn perform_randomized_svd_for_loadings(
        matrix_features_by_samples: &ArrayView2<f32>,
        num_components_target_k: usize,
        sketch_oversampling_count: usize,
        num_power_iterations: usize,
        random_seed: u64,
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let (u_opt, _s_opt, _v_opt) = Self::_internal_perform_rsvd(
            matrix_features_by_samples,
            num_components_target_k,
            sketch_oversampling_count,
            num_power_iterations,
            random_seed,
            true,  // request_u_components
            false, // request_s_components
            false, // request_v_components
        )?;

        u_opt.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Loadings (U matrix) not computed or available from _internal_perform_rsvd",
            )) as ThreadSafeStdError
        })
    }
    
    fn compute_refined_snp_loadings<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        initial_sample_pc_scores: &InitialSamplePcScores,
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let initial_scores_n_by_k_initial = &initial_sample_pc_scores.scores; 
        let num_qc_samples = initial_scores_n_by_k_initial.nrows();
        let num_computed_initial_pcs = initial_scores_n_by_k_initial.ncols();
        let num_total_pca_snps = genotype_data.num_pca_snps();

        info!(
            "Refining SNP loadings ({} SNPs, {} initial PCs from {} samples).",
            num_total_pca_snps, num_computed_initial_pcs, num_qc_samples
        );

        if num_computed_initial_pcs == 0 {
            debug!("No initial PCs to refine loadings for, returning empty loadings matrix.");
            return Ok(Array2::zeros((num_total_pca_snps, 0)));
        }
        if num_total_pca_snps == 0 {
            debug!("No PCA SNPs available, returning empty loadings matrix.");
            return Ok(Array2::zeros((0, num_computed_initial_pcs)));
        }

        let mut snp_loadings_before_ortho_pca_snps_by_components = 
            Array2::<f32>::zeros((num_total_pca_snps, num_computed_initial_pcs));
        let all_qc_sample_ids: Vec<QcSampleId> = (0..num_qc_samples).map(QcSampleId).collect();
        
        let snp_processing_strip_size = 2000.min(num_total_pca_snps).max(1);
        
        if snp_processing_strip_size > 0 {
            snp_loadings_before_ortho_pca_snps_by_components
                .axis_chunks_iter_mut(Axis(0), snp_processing_strip_size)
                .into_par_iter()
                .enumerate()
                .try_for_each(|(strip_index, mut loadings_strip_view_mut)| 
                    -> Result<(), ThreadSafeStdError> {
                    let strip_start_snp_idx = strip_index * snp_processing_strip_size;
                    let num_snps_in_current_strip = loadings_strip_view_mut.nrows();

                    let snp_ids_in_strip: Vec<PcaSnpId> = (strip_start_snp_idx..strip_start_snp_idx + num_snps_in_current_strip)
                        .map(PcaSnpId)
                        .collect();

                    if snp_ids_in_strip.is_empty() { return Ok(()); }

                    let genotype_data_strip_snps_by_samples = genotype_data.get_standardized_snp_sample_block(
                        &snp_ids_in_strip,
                        &all_qc_sample_ids,
                    ).map_err(|e_accessor| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get standardized SNP/sample block during refined SNP loading for strip index {}: {}", strip_index, e_accessor))) as ThreadSafeStdError)?;
                        
                    let snp_loadings_for_strip = genotype_data_strip_snps_by_samples.dot(initial_scores_n_by_k_initial);
                    loadings_strip_view_mut.assign(&snp_loadings_for_strip);
                    Ok(())
                })?;
        }
        
        if snp_loadings_before_ortho_pca_snps_by_components.ncols() == 0 {
            info!("Refined loadings matrix has 0 columns, QR skipped.");
            return Ok(snp_loadings_before_ortho_pca_snps_by_components);
        }
        
        let backend = LinAlgBackendProvider::<f32>::new(); // Use LinAlgBackendProvider for f32
        let orthonormal_snp_loadings = backend.qr_q_factor(&snp_loadings_before_ortho_pca_snps_by_components)
            .map_err(|e_qr| -> ThreadSafeStdError {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("QR decomposition of refined loadings failed (via backend): {}", e_qr)
                ).into()
            })?;
        
        info!("Computed refined SNP loadings. Shape: {:?}", orthonormal_snp_loadings.dim());
        Ok(orthonormal_snp_loadings)
    }

    fn compute_final_scores_and_eigenvalues<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        orthonormal_snp_loadings_snps_by_components: &ArrayView2<f32>, 
        num_total_qc_samples: usize,
    ) -> Result<(Array2<f32>, Array1<f64>), ThreadSafeStdError> {
        let num_total_pca_snps = orthonormal_snp_loadings_snps_by_components.nrows();
        let num_final_computed_pcs = orthonormal_snp_loadings_snps_by_components.ncols();

        info!(
            "Computing final sample scores ({} samples, {} PCs) and eigenvalues.",
            num_total_qc_samples, num_final_computed_pcs
        );

        if num_final_computed_pcs == 0 {
            debug!("No final PCs to compute scores/eigenvalues for, returning empty results.");
            return Ok((Array2::zeros((num_total_qc_samples, 0)), Array1::zeros(0)));
        }
        if num_total_pca_snps == 0 {
            debug!("No PCA SNPs available for final scores, returning empty results for {} PCs.", num_final_computed_pcs);
            return Ok((Array2::zeros((num_total_qc_samples, num_final_computed_pcs)), Array1::zeros(num_final_computed_pcs)));
        }
        if num_total_qc_samples == 0 {
            debug!("No QC samples available for final scores, returning empty results for {} PCs.", num_final_computed_pcs);
            return Ok((Array2::zeros((0, num_final_computed_pcs)), Array1::zeros(num_final_computed_pcs)));
        }

        let snp_processing_strip_size = 2000.min(num_total_pca_snps).max(1);
        let all_qc_sample_ids_for_final_scores: Vec<QcSampleId> =
            (0..num_total_qc_samples).map(QcSampleId).collect();

        // Define SNP strip iteration parameters
        let strip_indices_starts: Vec<usize> = (0..num_total_pca_snps)
            .step_by(snp_processing_strip_size)
            .collect();
        // Parallel computation of score contributions from each SNP strip
        let strip_score_contributions_results: Result<Vec<Array2<f32>>, ThreadSafeStdError> = strip_indices_starts
            .par_iter()
            .map(|&strip_start_snp_idx| -> Result<Array2<f32>, ThreadSafeStdError> {
                let strip_end_snp_idx = (strip_start_snp_idx + snp_processing_strip_size).min(num_total_pca_snps);
                
                // If the strip has no SNPs, return zeros.
                if strip_start_snp_idx >= strip_end_snp_idx {
                    return Ok(Array2::<f32>::zeros((num_total_qc_samples, num_final_computed_pcs)));
                }

                let snp_ids_in_strip_as_pca_ids: Vec<PcaSnpId> =
                    (strip_start_snp_idx..strip_end_snp_idx).map(PcaSnpId).collect();

                let genotype_data_strip_snps_by_samples = genotype_data.get_standardized_snp_sample_block(
                    &snp_ids_in_strip_as_pca_ids,
                    &all_qc_sample_ids_for_final_scores,
                ).map_err(|e_accessor| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get standardized SNP/sample block during final score computation for strip starting at {}: {}", strip_start_snp_idx, e_accessor))) as ThreadSafeStdError)?; // M_strip x N
                
                let loadings_for_strip_snps_by_components = orthonormal_snp_loadings_snps_by_components
                    .slice(s![strip_start_snp_idx..strip_end_snp_idx, ..]); // M_strip x K
                
                // S_final_strip = X_strip^T V_final_strip
                // (M_strip x N)^T dot (M_strip x K) -> (N x M_strip) dot (M_strip x K) -> N x K
                let scores_contribution_from_strip_n_by_k =
                    genotype_data_strip_snps_by_samples.t().dot(&loadings_for_strip_snps_by_components);
                
                Ok(scores_contribution_from_strip_n_by_k)
            })
            .collect(); // Collects into Result<Vec<Array2<f32>>, ThreadSafeStdError>

        // Sum the contributions. If any map failed, propagate the error from collecting results.
        // Otherwise, reduce the vector of matrices.
        let computed_final_sample_scores_samples_by_components: Array2<f32> = strip_score_contributions_results? // Yields Vec<Array2<f32>> or propagates error
            .into_par_iter() // Convert Vec<Array2<f32>> to ParIter<Item = Array2<f32>>
            .reduce( // Reduce on ParIter. Returns Self::Item (i.e., Array2<f32>).
                // The identity function provides the initial value for reduction tasks,
                // and is returned if the iterator is empty.
                || Array2::<f32>::zeros((num_total_qc_samples, num_final_computed_pcs)), 
                |mut acc_matrix, strip_scores_matrix| {
                    acc_matrix += &strip_scores_matrix; // Sum contributions
                    acc_matrix
                },
            ); // The result of reduce is directly Array2<f32>.

        let mut computed_final_pc_eigenvalues = Array1::<f64>::zeros(num_final_computed_pcs);
        if num_total_qc_samples > 1 {
            for k_idx in 0..num_final_computed_pcs {
                let pc_scores_column_f64 = computed_final_sample_scores_samples_by_components
                    .column(k_idx)
                    .mapv(|val| val as f64);
                
                let sum_of_squares: f64 = pc_scores_column_f64.iter().map(|&val| val.powi(2)).sum();
                computed_final_pc_eigenvalues[k_idx] = sum_of_squares / (num_total_qc_samples as f64 - 1.0);
            }
        } else if num_total_qc_samples == 1 && num_final_computed_pcs > 0 { 
            computed_final_pc_eigenvalues.fill(0.0);
        }

        debug!("Computed final eigenvalues: {:?}", computed_final_pc_eigenvalues);
        info!("Computed final sample scores. Shape: {:?}", computed_final_sample_scores_samples_by_components.dim());
        Ok((computed_final_sample_scores_samples_by_components, computed_final_pc_eigenvalues))
    }

    /// Performs randomized SVD on a matrix A (matrix_features_by_samples, M x N).
    /// Returns (Option<U_A_approx>, Option<S_A_approx>, Option<V_A_approx>)
    /// U_A_approx: M x K_eff (Left singular vectors of A)
    /// S_A_approx: K_eff (Singular values of A)
    /// V_A_approx: N x K_eff (Right singular vectors of A)
    #[allow(clippy::too_many_arguments)]
    fn _internal_perform_rsvd(
        matrix_features_by_samples: &ArrayView2<f32>, // Input matrix A (M features x N samples)
        num_components_target_k: usize,              // Desired K
        sketch_oversampling_count: usize,            // p (for L = K+p)
        num_power_iterations: usize,                 // q
        random_seed: u64,
        request_u_components: bool, // True if U (left singular vectors) is needed
        request_s_components: bool, // True if S (singular values) is needed
        request_v_components: bool  // True if V (right singular vectors) is needed
    ) -> Result<(Option<Array2<f32>>, Option<Array1<f32>>, Option<Array2<f32>>), ThreadSafeStdError> {
        let num_features_m = matrix_features_by_samples.nrows();
        let num_samples_n = matrix_features_by_samples.ncols();

        if num_features_m == 0 || num_samples_n == 0 || num_components_target_k == 0 {
            debug!("RSVD: Input matrix empty or K=0. M={}, N={}, K={}", num_features_m, num_samples_n, num_components_target_k);
            let u_res = if request_u_components { Some(Array2::zeros((num_features_m, 0))) } else { None };
            let s_res = if request_s_components { Some(Array1::zeros(0)) } else { None };
            let v_res = if request_v_components { Some(Array2::zeros((num_samples_n, 0))) } else { None };
            return Ok((u_res, s_res, v_res));
        }

        let sketch_dimension_l = (num_components_target_k + sketch_oversampling_count)
            .min(num_features_m.min(num_samples_n));

        if sketch_dimension_l == 0 {
            debug!("RSVD: Sketch dimension L=0. M={}, N={}, K={}, p={}", num_features_m, num_samples_n, num_components_target_k, sketch_oversampling_count);
            let u_res = if request_u_components { Some(Array2::zeros((num_features_m, 0))) } else { None };
            let s_res = if request_s_components { Some(Array1::zeros(0)) } else { None };
            let v_res = if request_v_components { Some(Array2::zeros((num_samples_n, 0))) } else { None };
            return Ok((u_res, s_res, v_res));
        }
        trace!(
            "RSVD internal: Target_K={}, Sketch_L={}, Input_M(features)={}, Input_N(samples)={}",
            num_components_target_k, sketch_dimension_l, num_features_m, num_samples_n
        );

        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let normal_dist = Normal::new(0.0, 1.0)
            .map_err(|e_normal| -> ThreadSafeStdError {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to create normal distribution for RSVD: {}", e_normal),
                ).into()
            })?;
        
        // Omega: N x L
        let random_projection_matrix_omega = Array2::from_shape_fn((num_samples_n, sketch_dimension_l), |_| {
            normal_dist.sample(&mut rng) as f32
        });
        
        let backend = LinAlgBackendProvider::<f32>::new();
        
        // Y = A * Omega (M x N) * (N x L) -> (M x L)
        let sketch_y = matrix_features_by_samples.dot(&random_projection_matrix_omega);

        if sketch_y.ncols() == 0 {
            warn!("RSVD: Initial sketch Y (A*Omega) has zero columns before first QR. Target_K={}, Sketch_L={}", num_components_target_k, sketch_dimension_l);
            let u_res = if request_u_components { Some(Array2::zeros((num_features_m, 0))) } else { None };
            let s_res = if request_s_components { Some(Array1::zeros(0)) } else { None };
            let v_res = if request_v_components { Some(Array2::zeros((num_samples_n, 0))) } else { None };
            return Ok((u_res, s_res, v_res));
        }
        
        // Q_basis = orth(Y) (M x L_actual_y)
        let mut q_basis_m_by_l_actual = backend.qr_q_factor(&sketch_y)
            .map_err(|e_qr| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("QR decomposition of initial sketch Y failed in RSVD: {}", e_qr))) as ThreadSafeStdError)?;
        
        // Power iterations
        for iter_idx in 0..num_power_iterations {
            if q_basis_m_by_l_actual.ncols() == 0 { 
                trace!("RSVD Power Iteration {}: Q_basis became empty, breaking.", iter_idx + 1);
                break; 
            }
            trace!("RSVD Power Iteration {}/{}", iter_idx + 1, num_power_iterations);
            
            // Q_tilde_candidate = A.T * Q_basis (N x M) * (M x L_actual) -> (N x L_actual)
            let q_tilde_candidate = matrix_features_by_samples.t().dot(&q_basis_m_by_l_actual);
            if q_tilde_candidate.ncols() == 0 { 
                q_basis_m_by_l_actual = Array2::zeros((q_basis_m_by_l_actual.nrows(),0)); 
                trace!("RSVD Power Iteration {}: Q_tilde_candidate became empty.", iter_idx + 1);
                break; 
            }
            // Q_tilde = orth(Q_tilde_candidate) (N x L_actual_tilde)
            let q_tilde_n_by_l_actual = backend.qr_q_factor(&q_tilde_candidate)
                .map_err(|e_qr| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("QR for Q_tilde in power iteration {} failed: {}", iter_idx + 1, e_qr))) as ThreadSafeStdError)?;

            if q_tilde_n_by_l_actual.ncols() == 0 {
                q_basis_m_by_l_actual = Array2::zeros((q_basis_m_by_l_actual.nrows(),0));
                trace!("RSVD Power Iteration {}: Q_tilde became empty after QR.", iter_idx + 1);
                break;
            }

            // Q_basis_candidate = A * Q_tilde (M x N) * (N x L_actual_tilde) -> (M x L_actual_tilde)
            let q_basis_candidate_next = matrix_features_by_samples.dot(&q_tilde_n_by_l_actual);
            if q_basis_candidate_next.ncols() == 0 {
                 q_basis_m_by_l_actual = Array2::zeros((q_basis_m_by_l_actual.nrows(),0));
                 trace!("RSVD Power Iteration {}: Q_basis_candidate_next became empty.", iter_idx + 1);
                 break;
            }
            // Q_basis = orth(Q_basis_candidate_next) (M x L_actual_final_iter)
            q_basis_m_by_l_actual = backend.qr_q_factor(&q_basis_candidate_next)
                .map_err(|e_qr| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("QR for Q_basis in power iteration {} failed: {}", iter_idx + 1, e_qr))) as ThreadSafeStdError)?;
        }
        
        if q_basis_m_by_l_actual.ncols() == 0 {
            warn!("RSVD: Refined Q_basis has zero columns after power iterations. Target_K={}", num_components_target_k);
            let u_res = if request_u_components { Some(Array2::zeros((num_features_m, 0))) } else { None };
            let s_res = if request_s_components { Some(Array1::zeros(0)) } else { None };
            let v_res = if request_v_components { Some(Array2::zeros((num_samples_n, 0))) } else { None };
            return Ok((u_res, s_res, v_res));
        }
        
        // B = Q_basis.T * A (L_actual x M) * (M x N) -> (L_actual x N)
        let projected_b_l_actual_by_n = q_basis_m_by_l_actual.t().dot(matrix_features_by_samples);
        
        // SVD of B: B = U_B * S_B * V_B.T
        // U_B is L_actual x rank_b
        // S_B is rank_b
        // V_B.T is rank_b x N
        let compute_u_for_b = request_u_components; // U_A = Q_basis * U_B, so U_B is needed if U_A is.
        let compute_v_for_b = request_v_components; // V_A = V_B, so V_B (from V_B.T) is needed if V_A is.

        let svd_output_b = backend.svd_into(projected_b_l_actual_by_n.into_owned(), compute_u_for_b, compute_v_for_b)
            .map_err(|e_svd| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("SVD of projected matrix B failed in RSVD: {}", e_svd))) as ThreadSafeStdError)?;
        
        let mut u_a_approx_opt: Option<Array2<f32>> = None;
        let mut s_a_approx_opt: Option<Array1<f32>> = None;
        let mut v_a_approx_opt: Option<Array2<f32>> = None;

        let effective_rank_b = svd_output_b.s.as_ref().map_or(0, |s_vals| s_vals.len());
        let num_k_to_return = num_components_target_k.min(effective_rank_b);

        if request_s_components {
            if let Some(s_b_values) = svd_output_b.s {
                s_a_approx_opt = Some(s_b_values.slice(s![0..num_k_to_return]).to_owned());
            } else {
                // Should not happen if svd_into succeeds and effective_rank_b > 0, but handle defensively
                s_a_approx_opt = Some(Array1::zeros(0));
            }
        }

        if request_u_components {
            if let Some(u_b_l_actual_by_rank_b) = svd_output_b.u {
                if u_b_l_actual_by_rank_b.ncols() > 0 && q_basis_m_by_l_actual.ncols() > 0 {
                    // U_A = Q_basis * U_B (M x L_actual) * (L_actual x rank_b) -> M x rank_b
                    let u_a_approx_m_by_rank_b = q_basis_m_by_l_actual.dot(&u_b_l_actual_by_rank_b);
                    u_a_approx_opt = Some(u_a_approx_m_by_rank_b.slice_axis(Axis(1), s![0..num_k_to_return]).to_owned());
                } else {
                     u_a_approx_opt = Some(Array2::zeros((num_features_m, 0)));   
                }
            } else {
                u_a_approx_opt = Some(Array2::zeros((num_features_m, 0)));
            }
        }

        if request_v_components {
            if let Some(v_b_t_rank_b_by_n) = svd_output_b.vt {
                if v_b_t_rank_b_by_n.nrows() > 0 { // effectively checks rank_b > 0
                    // V_A = V_B. V_B is (N x rank_b). We have V_B.T (rank_b x N)
                    let v_a_approx_n_by_rank_b = v_b_t_rank_b_by_n.t().into_owned();
                    v_a_approx_opt = Some(v_a_approx_n_by_rank_b.slice_axis(Axis(1), s![0..num_k_to_return]).to_owned());
                } else {
                    v_a_approx_opt = Some(Array2::zeros((num_samples_n, 0)));
                }
            } else {
                 v_a_approx_opt = Some(Array2::zeros((num_samples_n, 0)));
            }
        }
        
        trace!(
            "RSVD internal successfully computed components. U_shape={:?}, S_len={}, V_shape={:?}",
            u_a_approx_opt.as_ref().map(|m| m.dim()),
            s_a_approx_opt.as_ref().map_or(0, |s| s.len()),
            v_a_approx_opt.as_ref().map(|m| m.dim())
        );
        Ok((u_a_approx_opt, s_a_approx_opt, v_a_approx_opt))
    }
}
