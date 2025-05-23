use ndarray::{s, Array1, Array2, Axis, ArrayView1, ArrayView2};
use ndarray_linalg::{Eigh, UPLO, QR, SVDInto, Lapack};
use rayon::prelude::*;
use std::error::Error;
use log::{info, debug, trace, warn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

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
            random_seed: 2025,
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

    // --- Main Public Execution Method ---

    /// Orchestrates the entire EigenSNP PCA workflow.
    pub fn compute_pca<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specifications: &[LdBlockSpecification],
    ) -> Result<EigenSNPCoreOutput, Box<dyn Error>> {
        let num_total_qc_samples = genotype_data.num_qc_samples();
        let num_total_pca_snps = genotype_data.num_pca_snps();

        // Determine subset sample IDs based on config
        let desired_ns = (self.config.subset_factor_for_local_basis_learning * num_total_qc_samples as f64).round() as usize;
        let ns_clamped_min = desired_ns.max(self.config.min_subset_size_for_local_basis_learning);
        let actual_ns = ns_clamped_min.min(self.config.max_subset_size_for_local_basis_learning).min(num_total_qc_samples);

        info!(
            "Starting EigenSNP PCA. Target PCs={}, Total Samples={}, Subset Samples (N_s)={}, Num LD Blocks={}",
            self.config.target_num_global_pcs,
            num_total_qc_samples,
            actual_ns,
            ld_block_specifications.len()
        );
        let overall_start_time = std::time::Instant::now();

        // Input Validations
        if self.config.target_num_global_pcs == 0 {
            return Err(Box::from("Target number of global PCs must be greater than 0."));
        }
        if num_total_pca_snps > 0 && ld_block_specifications.is_empty() {
            return Err(Box::from("LD block specifications cannot be empty if PCA SNPs are present."));
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

        let subset_sample_ids_selected: Vec<QcSampleId> = if actual_ns > 0 {
            let mut rng_subset_selection = ChaCha8Rng::seed_from_u64(self.config.random_seed);
            let all_sample_indices: Vec<usize> = (0..num_total_qc_samples).collect();
            let subset_indices: Vec<usize> = rand::seq::index::sample(&mut rng_subset_selection, num_total_qc_samples, actual_ns).into_vec();
            subset_indices.into_iter().map(QcSampleId).collect()
        } else {
             if num_total_qc_samples > 0 && ld_block_specifications.iter().any(|b| b.num_snps_in_block() > 0) {
                 warn!("Calculated N_s is 0, but total samples > 0 and blocks have SNPs. This situation is problematic for learning local bases.");
                 return Err(Box::from("Subset size (N_s) for local basis learning is 0, but samples and SNP blocks are present."));
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
            Self::standardize_rows_of_condensed_matrix(raw_condensed_feature_matrix)?;
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
    ) -> Result<Vec<PerBlockLocalSnpBasis>, Box<dyn Error>> {
        info!(
            "Learning local eigenSNP bases for {} LD blocks using N_subset = {} samples.",
            ld_block_specs.len(),
            subset_sample_ids.len()
        );

        if subset_sample_ids.is_empty() {
            let any_snps_in_blocks = ld_block_specs.iter().any(|b| b.num_snps_in_block() > 0);
            if any_snps_in_blocks {
                return Err(Box::from("Subset sample IDs for local basis learning cannot be empty if LD blocks contain SNPs."));
            }
        }

        let local_bases_results: Vec<Result<PerBlockLocalSnpBasis, Box<dyn Error>>> =
            ld_block_specs
                .par_iter()
                .enumerate()
                .map(|(block_idx_val, block_spec)| {
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
                        )?;
                    
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
                    
                    let genotype_block_f64 = genotype_block_for_subset_samples.mapv(|x_val| x_val as f64);
                    let snp_covariance_matrix_f64 = genotype_block_f64.dot(&genotype_block_f64.t());

                    let (_eigenvalues_f64, eigenvectors_f64_columns) =
                        snp_covariance_matrix_f64.eigh(UPLO::Upper)
                        .map_err(|e| Box::from(format!("Eigendecomposition failed for block ID {:?} ({}): {}", block_list_id, block_spec.user_defined_block_tag, e)))?;
                    
                    let selected_eigenvectors_f64 = eigenvectors_f64_columns.slice_axis(
                        Axis(1),
                        ndarray::Slice::from((actual_num_snps_in_block - num_components_to_extract)..actual_num_snps_in_block),
                    );

                    let local_basis_vectors_f32 = selected_eigenvectors_f64.mapv(|x_val| x_val as f32);

                    trace!("Block ID {:?} ({}): extracted {} local components.", block_list_id, block_spec.user_defined_block_tag, num_components_to_extract);
                    Ok(PerBlockLocalSnpBasis {
                        block_list_id,
                        basis_vectors: local_basis_vectors_f32.into_owned(),
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
    ) -> Result<RawCondensedFeatures, Box<dyn Error>> {
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
            let local_basis_data = &all_local_bases[block_idx]; // Assumes all_local_bases is ordered same as ld_block_specs
            
            let local_snp_basis_vectors = &local_basis_data.basis_vectors; 
            let num_components_this_block = local_snp_basis_vectors.ncols();

            if block_spec.num_snps_in_block() == 0 || num_components_this_block == 0 {
                trace!("Skipping block with tag '{}' for projection: num_snps={} or num_local_components=0.", block_spec.user_defined_block_tag, block_spec.num_snps_in_block());
                continue;
            }
            
            let genotype_data_for_block_all_samples = genotype_data.get_standardized_snp_sample_block(
                &block_spec.pca_snp_ids_in_block,
                &all_qc_sample_ids,
            )?;

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
    
    fn standardize_rows_of_condensed_matrix(
        raw_features_input: RawCondensedFeatures,
    ) -> Result<StandardizedCondensedFeatures, Box<dyn Error>> {
        let mut condensed_data_matrix = raw_features_input.data; 
        let num_total_condensed_features = condensed_data_matrix.nrows();
        let num_samples = condensed_data_matrix.ncols();

        info!(
            "Standardizing rows of condensed feature matrix ({} features, {} samples).",
            num_total_condensed_features, num_samples
        );

        if num_samples <= 1 { 
            if num_total_condensed_features > 0 && num_samples == 1 { 
                condensed_data_matrix.fill(0.0f32);
            } 
            debug!("Number of samples ({}) is <= 1 for condensed matrix; standardization may result in zeros or is skipped if empty.", num_samples);
            return Ok(StandardizedCondensedFeatures { data: condensed_data_matrix });
        }

        condensed_data_matrix
            .axis_iter_mut(Axis(0)) 
            .into_par_iter() 
            .for_each(|mut feature_row| {
                let mut sum_for_mean_f64: f64 = 0.0;
                for val_ref in feature_row.iter() {
                    sum_for_mean_f64 += *val_ref as f64;
                }
                let mean_val_f64 = sum_for_mean_f64 / (num_samples as f64);
                let mean_val_f32 = mean_val_f64 as f32;

                feature_row.mapv_inplace(|x_val| x_val - mean_val_f32);

                let mut sum_sq_deviations_f64: f64 = 0.0;
                for val_centered_ref in feature_row.iter() {
                    sum_sq_deviations_f64 += (*val_centered_ref as f64).powi(2);
                }
                
                let variance_f64 = sum_sq_deviations_f64 / (num_samples as f64 - 1.0); 
                let std_dev_f64 = variance_f64.sqrt();
                let std_dev_f32 = std_dev_f64 as f32;

                if std_dev_f32.abs() > 1e-7 { 
                    feature_row.mapv_inplace(|x_val| x_val / std_dev_f32);
                } else {
                    feature_row.fill(0.0f32);
                }
            });
        
        info!("Finished standardizing rows of condensed feature matrix.");
        Ok(StandardizedCondensedFeatures { data: condensed_data_matrix })
    }

    fn compute_pca_on_standardized_condensed_features_via_rsvd(
        &self,
        standardized_condensed_features: &StandardizedCondensedFeatures,
    ) -> Result<InitialSamplePcScores, Box<dyn Error>> {
        let sample_scores_n_by_k = Self::perform_randomized_svd_for_scores(
            &standardized_condensed_features.data.view(),
            self.config.target_num_global_pcs,
            self.config.global_pca_sketch_oversampling,
            self.config.global_pca_num_power_iterations,
            self.config.random_seed,
        )?;
        Ok(InitialSamplePcScores { scores: sample_scores_n_by_k })
    }

    fn perform_randomized_svd_for_scores(
        matrix_features_by_samples: &ArrayView2<f32>, 
        num_components_target_k: usize,
        sketch_oversampling_count: usize,
        num_power_iterations: usize,
        random_seed: u64,
    ) -> Result<Array2<f32>, Box<dyn Error>> {
        let num_features = matrix_features_by_samples.nrows();
        let num_samples = matrix_features_by_samples.ncols();

        if num_features == 0 || num_samples == 0 || num_components_target_k == 0 {
            return Ok(Array2::zeros((num_samples, 0)));
        }

        let sketch_dimension = (num_components_target_k + sketch_oversampling_count)
            .min(num_features.min(num_samples));

        if sketch_dimension == 0 {
            return Ok(Array2::zeros((num_samples, 0)));
        }
        trace!(
            "RSVD for scores: Target_K={}, Sketch_L={}, Input_Features={}, Input_Samples={}",
            num_components_target_k, sketch_dimension, num_features, num_samples
        );

        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let normal_dist = Normal::new(0.0, 1.0)
            .map_err(|e| Box::from(format!("Failed to create normal distribution for RSVD: {}", e)))?;

        let random_projection_matrix_samples_by_sketch = Array2::from_shape_fn((num_samples, sketch_dimension), |_| {
            normal_dist.sample(&mut rng) as f32
        });
        
        let mut orthonormal_basis_features_by_sketch = matrix_features_by_samples.dot(&random_projection_matrix_samples_by_sketch);

        if orthonormal_basis_features_by_sketch.ncols() == 0 {
            warn!("RSVD: Initial sketch Y (A*Omega) has zero columns before first QR. Target_K={}, Sketch_L={}", num_components_target_k, sketch_dimension);
            return Ok(Array2::zeros((num_samples, 0)));
        }
        orthonormal_basis_features_by_sketch = orthonormal_basis_features_by_sketch.qr()?.0;

        for iter_idx in 0..num_power_iterations {
            if orthonormal_basis_features_by_sketch.ncols() == 0 { break; } 
            trace!("RSVD Power Iteration {}/{}", iter_idx + 1, num_power_iterations);
            
            let projected_onto_samples = matrix_features_by_samples.t().dot(&orthonormal_basis_features_by_sketch); 
            if projected_onto_samples.ncols() == 0 { 
                orthonormal_basis_features_by_sketch = Array2::zeros((orthonormal_basis_features_by_sketch.nrows(),0)); 
                break; 
            }
            
            orthonormal_basis_features_by_sketch = matrix_features_by_samples.dot(&projected_onto_samples); 
            if orthonormal_basis_features_by_sketch.ncols() == 0 { break; }
            orthonormal_basis_features_by_sketch = orthonormal_basis_features_by_sketch.qr()?.0; 
        }

        if orthonormal_basis_features_by_sketch.ncols() == 0 {
            warn!("RSVD: Refined sketch Q_basis for left singular vectors has zero columns. Returning empty scores.");
            return Ok(Array2::zeros((num_samples, 0)));
        }
        
        let projected_onto_sketch_basis_sketch_dim_by_samples = orthonormal_basis_features_by_sketch.t().dot(matrix_features_by_samples);

        let svd_of_projected_b = projected_onto_sketch_basis_sketch_dim_by_samples.svd_into(false, true) 
            .map_err(|e| Box::from(format!("SVD of projected matrix B failed in RSVD: {}", e)))?;
        
        let right_singular_vectors_transposed_of_b = svd_of_projected_b.2
            .ok_or_else(|| Box::from("V^T from SVD of B was not computed in RSVD."))?;
        
        let computed_sample_scores_samples_by_rank_b = right_singular_vectors_transposed_of_b.t().into_owned();

        let num_components_to_return = num_components_target_k.min(computed_sample_scores_samples_by_rank_b.ncols());
        let final_sample_scores_n_by_k = computed_sample_scores_samples_by_rank_b.slice_axis(Axis(1), s![..num_components_to_return]).to_owned();
        
        trace!("RSVD successfully computed scores. Shape: {:?}", final_sample_scores_n_by_k.dim());
        Ok(final_sample_scores_n_by_k)
    }

    fn compute_refined_snp_loadings<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        initial_sample_pc_scores: &InitialSamplePcScores,
    ) -> Result<Array2<f32>, Box<dyn Error>> {
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
                    -> Result<(), Box<dyn Error>> {
                    let strip_start_snp_idx = strip_index * snp_processing_strip_size;
                    let num_snps_in_current_strip = loadings_strip_view_mut.nrows();

                    let snp_ids_in_strip: Vec<PcaSnpId> = (strip_start_snp_idx..strip_start_snp_idx + num_snps_in_current_strip)
                        .map(PcaSnpId)
                        .collect();

                    if snp_ids_in_strip.is_empty() { return Ok(()); }

                    let genotype_data_strip_snps_by_samples = genotype_data.get_standardized_snp_sample_block(
                        &snp_ids_in_strip,
                        &all_qc_sample_ids,
                    )?;
                    
                    let snp_loadings_for_strip = genotype_data_strip_snps_by_samples.dot(initial_scores_n_by_k_initial);
                    loadings_strip_view_mut.assign(&snp_loadings_for_strip);
                    Ok(())
                })?;
        }
        
        if snp_loadings_before_ortho_pca_snps_by_components.ncols() == 0 {
            info!("Refined loadings matrix has 0 columns, QR skipped.");
            return Ok(snp_loadings_before_ortho_pca_snps_by_components);
        }

        let (orthonormal_snp_loadings, _r_matrix_for_qr) = 
            snp_loadings_before_ortho_pca_snps_by_components.qr()
            .map_err(|e| Box::from(format!("QR decomposition of refined loadings failed: {}", e)))?;
        
        info!("Computed refined SNP loadings. Shape: {:?}", orthonormal_snp_loadings.dim());
        Ok(orthonormal_snp_loadings)
    }

    fn compute_final_scores_and_eigenvalues<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        orthonormal_snp_loadings_d_by_k: &ArrayView2<f32>, 
        num_total_qc_samples: usize,
    ) -> Result<(Array2<f32>, Array1<f64>), Box<dyn Error>> {
        let num_total_pca_snps = orthonormal_snp_loadings_d_by_k.nrows();
        let num_final_computed_pcs = orthonormal_snp_loadings_d_by_k.ncols();

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

        let mut computed_final_sample_scores_samples_by_components =
            Array2::<f32>::zeros((num_total_qc_samples, num_final_computed_pcs));
        
        let snp_processing_strip_size = 2000.min(num_total_pca_snps).max(1);
        let all_qc_sample_ids_for_block: Vec<QcSampleId> =
            (0..num_total_qc_samples).map(QcSampleId).collect();

        // Iterate through SNP strips ONCE
        for strip_start_snp_idx in (0..num_total_pca_snps).step_by(snp_processing_strip_size) {
            let strip_end_snp_idx = (strip_start_snp_idx + snp_processing_strip_size).min(num_total_pca_snps);
            let snp_ids_in_strip_as_indices: Vec<usize> = (strip_start_snp_idx..strip_end_snp_idx).collect();
            let snp_ids_in_strip_as_pca_ids: Vec<PcaSnpId> =
                snp_ids_in_strip_as_indices.iter().map(|&idx| PcaSnpId(idx)).collect();

            if snp_ids_in_strip_as_pca_ids.is_empty() { continue; }

            let genotype_data_strip_snps_by_samples = genotype_data.get_standardized_snp_sample_block(
                &snp_ids_in_strip_as_pca_ids,
                &all_qc_sample_ids_for_block,
            )?; // M_strip x N

            let loadings_for_strip_snps_by_components = orthonormal_snp_loadings_d_by_k
                .slice(s![strip_start_snp_idx..strip_end_snp_idx, ..]); // M_strip x K
            
            // S_final = X V_final
            let scores_contribution_from_strip_n_by_k =
                genotype_data_strip_snps_by_samples.t().dot(&loadings_for_strip_snps_by_components);
            
            computed_final_sample_scores_samples_by_components += &scores_contribution_from_strip_n_by_k;
        }

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
}
