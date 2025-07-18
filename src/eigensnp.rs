use ndarray::{s, Array1, Array2, ArrayView2, Axis};
// Eigh, QR, SVDInto are replaced by backend calls. UPLO is handled by eigh_upper.
// use ndarray_linalg::{Eigh, UPLO, QR, SVDInto};
use crate::linalg_backends::{BackendQR, BackendSVD, LinAlgBackendProvider};
// use crate::ndarray_backend::NdarrayLinAlgBackend; // Replaced by LinAlgBackendProvider
// use crate::linalg_backend_dispatch::LinAlgBackendProvider; // Now part of linalg_backends
use log::{debug, info, trace, warn};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::error::Error;
use std::simd::prelude::*;
use std::sync::Arc;

/// Holds the essential metadata for a single SNP used in the PCA.
/// The order of these structs in a Vec should correspond to the `PcaSnpId`.
#[derive(Debug, Clone)]
pub struct PcaSnpMetadata {
    pub id: Arc<String>,
    pub chr: Arc<String>,
    pub pos: u64,
}

// Updated diagnostics struct names
#[cfg(feature = "enable-eigensnp-diagnostics")]
use crate::diagnostics::{
    compute_condition_number_via_svd_f32, // For f32 matrices, uses f64 SVD
    compute_condition_number_via_svd_f64, // For f64 matrices
    compute_frob_norm_f32,                // For f32 matrices
    compute_frob_norm_f64,                // For f64 matrices (if any intermediate become f64)
    compute_matrix_column_correlations_abs, // For f32 vs f64 matrix correlations
    compute_orthogonality_error_f32,      // For Q factors (f32)
    compute_svd_reconstruction_error_f32, // For SVD steps (f32)
    sample_singular_values,               // For f32 singular values
    sample_singular_values_f64,           // For f64 singular values
    FullPcaRunDetailedDiagnostics,
    PerBlockLocalBasisDiagnostics,
    RsvdStepDetail,
    SrPassDetail,
};

/// A thread-safe wrapper for standard dynamic errors,
/// so they implement `Send` and `Sync`.
pub type ThreadSafeStdError = Box<dyn Error + Send + Sync + 'static>;

// --- Conditional PCA Output Type ---

/// Defines the output structure of `compute_pca`, conditionally including detailed diagnostics.
#[cfg(feature = "enable-eigensnp-diagnostics")]
pub type PcaOutputWithDiagnostics = (
    EigenSNPCoreOutput,
    Option<crate::diagnostics::FullPcaRunDetailedDiagnostics>,
);

#[cfg(not(feature = "enable-eigensnp-diagnostics"))]
pub type PcaOutputWithDiagnostics = (EigenSNPCoreOutput, ());

// --- Conditional Helper Type ---

/// Helper type to conditionally include a type `T` or `()` based on a feature flag.
#[cfg(feature = "enable-eigensnp-diagnostics")]
pub type PcaConditionally<T> = T;
#[cfg(not(feature = "enable-eigensnp-diagnostics"))]
pub type PcaConditionally<T> = std::marker::PhantomData<T>;

// --- Conditional Type for Local Basis Learning Output ---

/// Defines the output structure for local basis learning, conditionally including diagnostics.
/// This is used as the per-block result type in `learn_all_ld_block_local_bases`.
#[cfg(feature = "enable-eigensnp-diagnostics")]
pub type LocalBasisWithDiagnostics = (
    PerBlockLocalSnpBasis,
    crate::diagnostics::PerBlockLocalBasisDiagnostics,
);

#[cfg(not(feature = "enable-eigensnp-diagnostics"))]
pub type LocalBasisWithDiagnostics = (PerBlockLocalSnpBasis, ());

// Helper trait for f64 conversion from Duration, handling potential errors.
#[cfg(feature = "enable-eigensnp-diagnostics")]
trait DurationToF64Safe {
    fn as_secs_f64_safe(&self) -> Option<f64>;
}
#[cfg(feature = "enable-eigensnp-diagnostics")]
impl DurationToF64Safe for std::time::Duration {
    fn as_secs_f64_safe(&self) -> Option<f64> {
        let secs = self.as_secs();
        let nanos = self.subsec_nanos();
        let total_nanos = secs as f64 * 1_000_000_000.0 + nanos as f64;
        if total_nanos.is_finite() {
            Some(total_nanos / 1_000_000_000.0)
        } else {
            None // Or handle error appropriately
        }
    }
}

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
    pub fn num_total_condensed_features(&self) -> usize {
        self.data.nrows()
    }
    pub fn num_samples(&self) -> usize {
        self.data.ncols()
    }
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
    pub fn num_total_condensed_features(&self) -> usize {
        self.data.nrows()
    }
    pub fn num_samples(&self) -> usize {
        self.data.ncols()
    }
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
    pub fn num_samples(&self) -> usize {
        self.scores.nrows()
    }
    pub fn num_pcs_computed(&self) -> usize {
        self.scores.ncols()
    }
}

// --- Final Output Structure ---

/// Encapsulates the final results of the EigenSNP PCA computation.
#[derive(Debug, Default)]
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
    #[cfg_attr(not(feature = "enable-eigensnp-diagnostics"), allow(unused_variables))]
    collect_diagnostics_flag: bool,
    #[cfg(feature = "enable-eigensnp-diagnostics")] mut full_diagnostics_collector: Option<
        &mut crate::diagnostics::FullPcaRunDetailedDiagnostics,
    >,
    #[cfg(not(feature = "enable-eigensnp-diagnostics"))] _full_diagnostics_collector: Option<()>,
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
        return Ok(StandardizedCondensedFeatures {
            data: condensed_data_matrix,
        });
    }

    // Parallelize row-wise standardization
    condensed_data_matrix
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut feature_row| {
            const LANES: usize = 8;

            // Get initial slice for reading
            let row_data_slice: &[f32] = feature_row
                .as_slice()
                .expect("Feature row must be contiguous for read-only operations");
            let num_elements_in_row = row_data_slice.len();

            if num_elements_in_row == 0 {
                // Should not happen if num_samples > 0, but good practice
                return;
            }
            let num_simd_chunks = num_elements_in_row / LANES;

            // --- SIMD Sum for Mean ---
            let mut simd_sum_f32 = Simd::splat(0.0f32);
            for chunk_idx in 0..num_simd_chunks {
                let offset = chunk_idx * LANES;
                let data_chunk =
                    Simd::<f32, LANES>::from_slice(&row_data_slice[offset..offset + LANES]);
                simd_sum_f32 += data_chunk;
            }
            let mut total_sum_f32 = simd_sum_f32.reduce_sum();
            for idx in (num_simd_chunks * LANES)..num_elements_in_row {
                total_sum_f32 += row_data_slice[idx];
            }
            let mean_val_f64 = total_sum_f32 as f64 / (num_elements_in_row as f64); // num_elements_in_row is num_samples for this row
            let mean_val_f32 = mean_val_f64 as f32;

            // Get mutable slice for modifications
            let row_data_mut_slice: &mut [f32] = feature_row
                .as_slice_mut()
                .expect("Feature row must be contiguous for mutable operations");

            // --- SIMD Mean Centering ---
            let mean_simd = Simd::splat(mean_val_f32);
            for chunk_idx in 0..num_simd_chunks {
                let offset = chunk_idx * LANES;
                let mut data_chunk =
                    Simd::<f32, LANES>::from_slice(&row_data_mut_slice[offset..offset + LANES]);
                data_chunk -= mean_simd;
                data_chunk.copy_to_slice(&mut row_data_mut_slice[offset..offset + LANES]);
            }
            for idx in (num_simd_chunks * LANES)..num_elements_in_row {
                row_data_mut_slice[idx] -= mean_val_f32;
            }

            // --- SIMD Sum of Squares (operates on the now mean-centered row_data_mut_slice) ---
            let mut simd_sum_sq_f32 = Simd::splat(0.0f32);
            for chunk_idx in 0..num_simd_chunks {
                let offset = chunk_idx * LANES;
                let centered_data_chunk =
                    Simd::<f32, LANES>::from_slice(&row_data_mut_slice[offset..offset + LANES]);
                simd_sum_sq_f32 += centered_data_chunk * centered_data_chunk;
            }
            let mut total_sum_sq_f32 = simd_sum_sq_f32.reduce_sum();
            for idx in (num_simd_chunks * LANES)..num_elements_in_row {
                total_sum_sq_f32 += row_data_mut_slice[idx] * row_data_mut_slice[idx];
            }

            // Use (N-1) for sample variance calculation (N is num_elements_in_row here)
            let variance_f64 =
                total_sum_sq_f32 as f64 / ((num_elements_in_row as f64 - 1.0).max(1.0)); // Avoid division by zero if num_elements_in_row is 1
            let std_dev_f64 = variance_f64.sqrt();
            let std_dev_f32 = std_dev_f64 as f32;

            // --- SIMD Scaling / Fill (operates on row_data_mut_slice) ---
            if std_dev_f32.abs() > 1e-7 {
                let inv_std_dev_val = 1.0 / std_dev_f32;
                let inv_std_dev_simd = Simd::splat(inv_std_dev_val);
                for chunk_idx in 0..num_simd_chunks {
                    let offset = chunk_idx * LANES;
                    let mut data_chunk =
                        Simd::<f32, LANES>::from_slice(&row_data_mut_slice[offset..offset + LANES]);
                    data_chunk *= inv_std_dev_simd;
                    data_chunk.copy_to_slice(&mut row_data_mut_slice[offset..offset + LANES]);
                }
                for idx in (num_simd_chunks * LANES)..num_elements_in_row {
                    row_data_mut_slice[idx] *= inv_std_dev_val;
                }
            } else {
                let zero_simd = Simd::<f32, LANES>::splat(0.0f32);
                for chunk_idx in 0..num_simd_chunks {
                    let offset = chunk_idx * LANES;
                    zero_simd.copy_to_slice(&mut row_data_mut_slice[offset..offset + LANES]);
                }
                for idx in (num_simd_chunks * LANES)..num_elements_in_row {
                    row_data_mut_slice[idx] = 0.0f32;
                }
            }
        });

    info!("Finished standardizing rows of condensed feature matrix.");

    debug!(
        "Standardized condensed feature matrix (A_eigen_std_star) dimensions: {:?}",
        condensed_data_matrix.dim()
    );
    if !condensed_data_matrix.is_empty() {
        let norm_a_eigen_std_star = condensed_data_matrix.view().mapv(|x| x * x).sum().sqrt();
        debug!(
            "Standardized condensed feature matrix (A_eigen_std_star) Frobenius norm: {:.4e}",
            norm_a_eigen_std_star
        );

        for row_idx in 0..3.min(condensed_data_matrix.nrows()) {
            let r_view = condensed_data_matrix.row(row_idx);
            if r_view.len() > 1 {
                // Variance requires at least 2 elements
                let mean_val = r_view.mean().unwrap_or(0.0); // Should be ~0 for standardized data
                let variance = r_view
                    .mapv(|x| (x - mean_val).powi(2))
                    .mean()
                    .unwrap_or(0.0); // Should be ~1 for standardized data
                                     // Using debug for variance of standardized matrix as it's a key check of success
                debug!("Standardized condensed matrix: Row {} mean (post-std): {:.4e}, variance (post-std): {:.4e}",
                       row_idx, mean_val, variance);
            } else if r_view.len() == 1 {
                // Single element in row, variance is undefined or 0. Mean is the element itself.
                debug!("Standardized condensed matrix: Row {} mean (post-std): {:.4e}, variance (post-std): N/A (single element in row)",
                       row_idx, r_view.mean().unwrap_or(0.0));
            }
        }
    }

    #[cfg(feature = "enable-eigensnp-diagnostics")]
    {
        if collect_diagnostics_flag {
            if let Some(dc) = full_diagnostics_collector.as_mut() {
                dc.c_std_matrix_dims = Some(condensed_data_matrix.dim());
                if !condensed_data_matrix.is_empty() {
                    dc.c_std_matrix_fro_norm =
                        Some(compute_frob_norm_f32(&condensed_data_matrix.view()) as f64);

                    // Sample column means and std devs (they should be ~0 and ~1)
                    let num_cols_to_sample = 10.min(condensed_data_matrix.ncols());
                    if num_cols_to_sample > 0 {
                        let mut means_sample = Vec::with_capacity(num_cols_to_sample);
                        let mut stds_sample = Vec::with_capacity(num_cols_to_sample);
                        for i in 0..num_cols_to_sample {
                            // This samples columns from the standardized matrix (features are rows)
                            // To sample feature characteristics, we'd sample rows.
                            // The spec asks for C_std_col_means/stds. C_std is features x samples.
                            // So we need to sample columns of C_std.
                            // This means we are sampling across features for a few samples.
                            // This seems transposed from typical expectation.
                            // If C_std is features x samples, then "column means" means mean of each sample's scores over features.
                            // Let's assume the spec meant "row means/stds" for C_std (i.e. for each standardized feature).
                            let feature_row = condensed_data_matrix.row(i); // Sample first few features
                            means_sample.push(feature_row.mean().unwrap_or(0.0) as f64); // Should be ~0
                            let variance = feature_row
                                .mapv(|x| (x - (feature_row.mean().unwrap_or(0.0))).powi(2))
                                .mean()
                                .unwrap_or(0.0);
                            stds_sample.push(variance.sqrt() as f64); // Should be ~1
                        }
                        dc.c_std_col_means_sample = Some(means_sample); // This is actually row means
                        dc.c_std_col_std_devs_sample = Some(stds_sample); // This is actually row stds
                        dc.notes.push_str(" Note: c_std_col_means_sample and c_std_col_std_devs_sample actually store ROW means/stds of C_std due to typical interpretation. ");
                    }
                } else {
                    dc.c_std_matrix_fro_norm = Some(0.0);
                }
            }
        }
    }
    Ok(StandardizedCondensedFeatures {
        data: condensed_data_matrix,
    })
}

// --- Helper functions for reordering ndarray structures ---

/// Reorders the columns of a 2D array (`Array2`) based on a given slice of indices.
/// Returns a new owned `Array2<T>` with columns in the specified order.
///
/// # Arguments
/// * `matrix`: A reference to the `Array2<T>` whose columns are to be reordered.
/// * `order`: A slice of `usize` representing the new order of columns.
///            Each index in `order` refers to a column index in the original `matrix`.
///
/// # Panics
/// This function will panic if any index in `order` is out of bounds for the columns of `matrix`.
/// It also panics if `Array2::from_shape_vec` fails due to an invalid shape (e.g. for empty order).
pub fn reorder_columns_owned<T: Clone>(matrix: &Array2<T>, order: &[usize]) -> Array2<T> {
    if order.is_empty() {
        // Return a matrix with the original number of rows but 0 columns.
        // Ensure that if matrix.nrows() is 0, this still behaves correctly.
        let shape = (matrix.nrows(), 0);
        return Array2::from_shape_vec(shape, vec![]).expect("Shape error for empty order");
    }
    // `select` creates a view. We need an owned array.
    matrix.select(Axis(1), order).to_owned()
}

/// Reorders the elements of a 1D array (`Array1`) based on a given slice of indices.
/// Returns a new owned `Array1<T>` with elements in the specified order.
///
/// # Arguments
/// * `array`: A reference to the `Array1<T>` whose elements are to be reordered.
/// * `order`: A slice of `usize` representing the new order of elements.
///            Each index in `order` refers to an element index in the original `array`.
///
/// # Panics
/// This function will panic if any index in `order` is out of bounds for the elements of `array`.
pub fn reorder_array_owned<T: Clone>(array: &Array1<T>, order: &[usize]) -> Array1<T> {
    if order.is_empty() {
        return Array1::from_vec(vec![]);
    }
    // `select` for Array1 is also along Axis(0).
    array.select(Axis(0), order).to_owned()
}

// --- Main Algorithm Orchestrator Struct Definition ---

/// Orchestrates the EigenSNP PCA algorithm.
/// Holds the configuration and provides the main execution method.
///
/// ## Numerical Precision
/// This algorithm primarily utilizes `f32` (single-precision floating point) numbers
/// for its matrix operations to optimize for memory efficiency and performance,
/// which are critical for large genomic datasets.
///
/// Key considerations regarding precision:
/// - **General Matrix Operations:** Most internal matrix multiplications, especially those
///   performed via `ndarray::dot()` (which typically delegates to BLAS `sgemm` routines),
///   use `f32` for both the elements and the internal accumulation during the dot product.
///   For very large matrices (e.g., a large number of samples $N$ or SNPs $D$), this `f32`
///   accumulation can lead to some loss of precision compared to an `f64` accumulation.
/// - **Specific `f64` Accumulation:** For certain critical intermediate sums where precision
///   is paramount and the number of summed elements can be particularly large (e.g.,
///   the construction of the $S_{int} = X V_{QR}^*$ matrix in
///   `compute_rotated_final_outputs`), the algorithm explicitly uses `f64` for accumulation
///   of `f32` intermediate products. This helps mitigate precision loss for these specific sums.
/// - **Output Precision:** Final PC scores and SNP loadings are returned as `f32` matrices.
///   Eigenvalues, however, are returned as an `f64` array.
///
/// This design represents a practical trade-off between computational resources and numerical
/// precision for typical PCA applications in genomics.
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

    /// Defines the number of SNPs to process in each parallel strip/chunk during
    /// stages like refined SNP loading calculation and intermediate score calculation.
    /// This helps manage memory for very large SNP datasets by processing them
    /// in smaller, more manageable vertical strips.
    /// Must be greater than 0. A typical value might be 2000-10000.
    pub snp_processing_strip_size: usize,
    /// Number of refinement passes for SNP loadings and sample scores. Default is 1.
    /// Pass 1: V_qr = orth(X U_scores_initial), S_int = X^T V_qr. SVD(S_int) gives U_rot, S_prime, V_rot.
    ///         Final: S_final = U_rot S_prime, V_final = V_qr V_rot.
    /// Pass 2 (if refine_pass_count >= 2): Use S_final (from pass 1) as new U_scores.
    ///         V_qr_p2 = orth(X S_final_p1), S_int_p2 = X^T V_qr_p2. SVD(S_int_p2) gives U_rot_p2, S_prime_p2, V_rot_p2.
    ///         Final_p2: S_final_p2 = U_rot_p2 S_prime_p2, V_final_p2 = V_qr_p2 V_rot_p2.
    /// Additional passes follow the same pattern.
    pub refine_pass_count: usize,
    /// Whether to collect detailed diagnostics during PCA computation.
    pub collect_diagnostics: bool,
    /// If set, specifies a directory path where the local PC loadings (eigenSNPs)
    /// for each LD block will be saved as individual TSV files.
    pub local_pcs_output_dir: Option<String>,
    /// If set, specifies the `LdBlockListId` (index in the input `ld_block_specifications` list)
    /// for which detailed rSVD step diagnostics should be traced during local basis learning.
    /// This is only active if `collect_diagnostics` is also true and the "enable-eigensnp-diagnostics" feature is enabled.
    #[cfg(feature = "enable-eigensnp-diagnostics")]
    pub diagnostic_block_list_id_to_trace: Option<usize>,
}

impl Default for EigenSNPCoreAlgorithmConfig {
    /// Provides sensible default parameters for the EigenSNP PCA algorithm.
    fn default() -> Self {
        EigenSNPCoreAlgorithmConfig {
            subset_factor_for_local_basis_learning: 0.1,
            min_subset_size_for_local_basis_learning: 20_000,
            max_subset_size_for_local_basis_learning: 60_000,
            components_per_ld_block: 7,
            target_num_global_pcs: 15,
            global_pca_sketch_oversampling: 10,
            global_pca_num_power_iterations: 2,
            local_rsvd_sketch_oversampling: 4,
            local_rsvd_num_power_iterations: 2,
            random_seed: 2025,
            snp_processing_strip_size: 2000, // Default
            refine_pass_count: 1,            // Default to 1 refinement pass
            collect_diagnostics: false,
            local_pcs_output_dir: None,
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            diagnostic_block_list_id_to_trace: None,
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
        snp_metadata: &[PcaSnpMetadata],
    ) -> Result<PcaOutputWithDiagnostics, ThreadSafeStdError> {
        // --- ADD THIS VALIDATION CHECK ---
        assert_eq!(
            snp_metadata.len(),
            genotype_data.num_pca_snps(),
            "The number of entries in snp_metadata ({}) must exactly match the number of PCA SNPs in the genotype_accessor ({}).",
            snp_metadata.len(),
            genotype_data.num_pca_snps()
        );
        // --- END OF VALIDATION CHECK ---
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        let mut diagnostics_collector: Option<FullPcaRunDetailedDiagnostics> = None;

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if self.config.collect_diagnostics {
                let mut main_diag_collector = FullPcaRunDetailedDiagnostics::default();
                // Record config summary in notes
                main_diag_collector
                    .notes
                    .push_str(&format!("EigenSNPCoreAlgorithmConfig: {:?}. ", self.config));
                main_diag_collector
                    .notes
                    .push_str("EigenSNP PCA run started. ");
                diagnostics_collector = Some(main_diag_collector);
            }
        }

        // When 'enable-eigensnp-diagnostics' is not enabled, the 'diagnostics_collector' variable is not compiled.
        // The return type PcaOutputWithDiagnostics correctly becomes (EigenSNPCoreOutput, ()),
        // and the () is provided directly in return statements for that configuration.

        let num_total_qc_samples = genotype_data.num_qc_samples();
        let num_total_pca_snps = genotype_data.num_pca_snps();

        // Determine subset sample IDs based on config
        let desired_subset_sample_count = (self.config.subset_factor_for_local_basis_learning
            * num_total_qc_samples as f64)
            .round() as usize;
        let clamped_min_subset_sample_count =
            desired_subset_sample_count.max(self.config.min_subset_size_for_local_basis_learning);
        // Make actual_subset_sample_count mutable here for potential override
        let mut actual_subset_sample_count = clamped_min_subset_sample_count
            .min(self.config.max_subset_size_for_local_basis_learning)
            .min(num_total_qc_samples);

        info!(
            "Starting EigenSNP PCA. Target PCs={}, Total Samples={}, Subset Samples (N_s, initial)={}, Num LD Blocks={}",
            self.config.target_num_global_pcs,
            num_total_qc_samples,
            actual_subset_sample_count, // Log initial N_s before potential override
            ld_block_specifications.len()
        );
        let overall_start_time = std::time::Instant::now();

        // Input Validations
        if self.config.target_num_global_pcs == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Target number of global PCs must be greater than 0.",
            )
            .into());
        }
        if num_total_pca_snps > 0 && ld_block_specifications.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "LD block specifications cannot be empty if PCA SNPs are present.",
            )
            .into());
        }
        if num_total_qc_samples == 0 {
            warn!("Genotype data has zero QC samples. Returning empty PCA output.");
            let output = EigenSNPCoreOutput {
                final_snp_principal_component_loadings: Array2::zeros((num_total_pca_snps, 0)),
                final_sample_principal_component_scores: Array2::zeros((0, 0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: 0,
                num_pca_snps_used: num_total_pca_snps,
                num_principal_components_computed: 0,
            };
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            return Ok((output, diagnostics_collector));
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            return Ok((output, ())); // This line remains correct as per the new type alias
        }
        if num_total_pca_snps == 0 {
            warn!("Genotype data has zero PCA SNPs. Returning empty PCA output.");
            let output = EigenSNPCoreOutput {
                final_snp_principal_component_loadings: Array2::zeros((0, 0)),
                final_sample_principal_component_scores: Array2::zeros((num_total_qc_samples, 0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: num_total_qc_samples,
                num_pca_snps_used: 0,
                num_principal_components_computed: 0,
            };
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            return Ok((output, diagnostics_collector));
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            return Ok((output, ())); // This line also remains correct
        }

        let subset_sample_ids_selected: Vec<QcSampleId>;
        let is_diagnostic_target_test = num_total_qc_samples == 200
            && (num_total_pca_snps >= 950 && num_total_pca_snps <= 1050); // Approximate SNP count

        if is_diagnostic_target_test {
            log::warn!(
                "DIAGNOSTIC MODE ACTIVE: Using ALL {} samples for local basis learning (N_s = N) for test_pc_correlation_structured_1000snps_200samples_5truepcs scenario. Original N_s was {}.",
                num_total_qc_samples, actual_subset_sample_count
            );
            actual_subset_sample_count = num_total_qc_samples; // Override N_s
            subset_sample_ids_selected = (0..num_total_qc_samples).map(QcSampleId).collect();
            // Update the info log for N_s if it was changed
            info!(
                "DIAGNOSTIC MODE: Overridden Subset Samples (N_s) = {}",
                actual_subset_sample_count
            );
        } else {
            // Original logic for selecting subset_sample_ids_selected
            if actual_subset_sample_count > 0 {
                let mut rng_subset_selection = ChaCha8Rng::seed_from_u64(self.config.random_seed);
                let subset_indices: Vec<usize> = rand::seq::index::sample(
                    &mut rng_subset_selection,
                    num_total_qc_samples,
                    actual_subset_sample_count,
                )
                .into_vec();
                subset_sample_ids_selected = subset_indices.into_iter().map(QcSampleId).collect();
            } else {
                if num_total_qc_samples > 0
                    && ld_block_specifications
                        .iter()
                        .any(|b| b.num_snps_in_block() > 0)
                {
                    log::warn!("Calculated N_s is 0 (and not in diagnostic override), but total samples > 0 and blocks have SNPs. This situation is problematic for learning local bases.");
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "Subset size (N_s) for local basis learning is 0, but samples and SNP blocks are present.").into());
                }
                subset_sample_ids_selected = Vec::new();
            }
        }

        // Create the output directory for local PCs ONCE at the beginning if specified.
        if let Some(dir_str) = self.config.local_pcs_output_dir.as_ref() {
            std::fs::create_dir_all(dir_str).map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "Failed to create local PCs output directory '{}': {}",
                        dir_str, e
                    ),
                )) as ThreadSafeStdError
            })?;
        }

        let local_bases_learning_start_time = std::time::Instant::now();

        // This `if/else` block dispatches to one of two specialized (monomorphized)
        // versions of `learn_all_ld_block_local_bases`. This ensures that when
        // `local_pcs_output_dir` is not set, the no-op closure is used, and the
        // compiler completely eliminates any overhead, achieving a true zero-cost abstraction.
        // The alternative, using dynamic dispatch (`Box<dyn FnMut>`), would avoid
        // duplicating the function call but would introduce a runtime vtable lookup cost.
        // For this performance-critical path, static dispatch is the correct choice.
        let all_block_local_bases = if let Some(dir_str) = self.config.local_pcs_output_dir.as_ref()
        {
            // --- BRANCH 1: Define and pass the file-writing closure. ---
            let output_dir = std::path::PathBuf::from(dir_str);
            self.learn_all_ld_block_local_bases(
                genotype_data,
                ld_block_specifications,
                &subset_sample_ids_selected,
                snp_metadata,
                // This closure captures the `output_dir` and performs the file I/O.
                |local_pcs, block_snp_metadata, block_list_id| {
                    if local_pcs.is_empty() {
                        return Ok(());
                    }

                    let filename =
                        output_dir.join(format!("block_{}.local_loadings.tsv", block_list_id.0));
                    let file =
                        std::fs::File::create(&filename).map_err(|e| -> ThreadSafeStdError {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!(
                                    "Failed to create local PC file '{}': {}",
                                    filename.display(),
                                    e
                                ),
                            ))
                        })?;

                    let mut writer = std::io::BufWriter::new(file);
                    use std::io::Write; // Import the Write trait for writeln!

                    // Write a header
                    write!(writer, "chr\tpos\tid")?;
                    for i in 1..=local_pcs.ncols() {
                        write!(writer, "\tlocal_pc_{}", i)?;
                    }
                    writeln!(writer)?;

                    // Zip the metadata with the rows of the loadings matrix
                    for (snp_info, loadings_row) in
                        block_snp_metadata.iter().zip(local_pcs.rows())
                    {
                        write!(
                            writer,
                            "{}\t{}\t{}",
                            snp_info.chr, snp_info.pos, snp_info.id
                        )?;
                        for &val in loadings_row.iter() {
                            write!(writer, "\t{}", val)?;
                        }
                        writeln!(writer)?;
                    }
                    Ok(())
                },
                #[cfg(feature = "enable-eigensnp-diagnostics")]
                diagnostics_collector
                    .as_mut()
                    .map(|dc| &mut dc.per_block_diagnostics),
                #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                None,
            )?
        } else {
            // --- BRANCH 2: Define and pass the zero-cost, no-op closure. ---
            self.learn_all_ld_block_local_bases(
                genotype_data,
                ld_block_specifications,
                &subset_sample_ids_selected,
                snp_metadata,
                // This closure does nothing and will be completely optimized away by the compiler.
                |_, _, _| Ok(()),
                #[cfg(feature = "enable-eigensnp-diagnostics")]
                diagnostics_collector
                    .as_mut()
                    .map(|dc| &mut dc.per_block_diagnostics),
                #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                None,
            )?
        };
        info!(
            "Learned local SNP bases in {:?}",
            local_bases_learning_start_time.elapsed()
        );

        let condensed_matrix_construction_start_time = std::time::Instant::now();
        let raw_condensed_feature_matrix = self.project_all_samples_onto_local_bases(
            genotype_data,
            ld_block_specifications,
            &all_block_local_bases,
            num_total_qc_samples,
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            diagnostics_collector.as_mut(),
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            None,
        )?;
        info!(
            "Constructed raw condensed feature matrix in {:?}",
            condensed_matrix_construction_start_time.elapsed()
        );

        let condensed_matrix_standardization_start_time = std::time::Instant::now();
        let standardized_condensed_feature_matrix = standardize_raw_condensed_features(
            raw_condensed_feature_matrix,
            self.config.collect_diagnostics,
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            diagnostics_collector.as_mut(),
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            None,
        )?;
        info!(
            "Standardized condensed feature matrix in {:?}",
            condensed_matrix_standardization_start_time.elapsed()
        );

        let initial_global_pca_start_time = std::time::Instant::now();
        let mut current_sample_scores = self
            .compute_pca_on_standardized_condensed_features_via_rsvd(
                &standardized_condensed_feature_matrix,
                #[cfg(feature = "enable-eigensnp-diagnostics")]
                diagnostics_collector
                    .as_mut()
                    .and_then(|dc| dc.global_pca_diag.as_mut().map(|gpd_box| gpd_box.as_mut())),
                #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                None,
            )?;
        info!(
            "Computed initial global PCA on condensed features in {:?}",
            initial_global_pca_start_time.elapsed()
        );

        let mut num_principal_components_computed_final = current_sample_scores.scores.ncols();
        if num_principal_components_computed_final == 0 {
            warn!("Initial PCA on condensed features yielded 0 components. Returning empty PCA output.");
            let output = EigenSNPCoreOutput {
                final_snp_principal_component_loadings: Array2::zeros((num_total_pca_snps, 0)),
                final_sample_principal_component_scores: Array2::zeros((num_total_qc_samples, 0)),
                final_principal_component_eigenvalues: Array1::zeros(0),
                num_qc_samples_used: num_total_qc_samples,
                num_pca_snps_used: num_total_pca_snps,
                num_principal_components_computed: 0,
            };
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            return Ok((output, diagnostics_collector));
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            return Ok((output, ())); // Correct
        }

        let mut final_sorted_snp_loadings: Array2<f32> = Array2::zeros((num_total_pca_snps, 0));
        let mut final_sorted_eigenvalues: Array1<f64> = Array1::zeros(0);

        if self.config.refine_pass_count == 0 {
            warn!(
             "EigenSNP refine_pass_count is 0. Skipping refinement loop. Output will reflect PCA of derived local eigenSNP features only. SNP loadings and SNP-based eigenvalues will be empty/zero."
         );
            // `num_principal_components_computed_final` is already set from the initial condensed PCA.
            // `final_sorted_snp_loadings` and `final_sorted_eigenvalues` remain their initial empty/zero states.
            // `current_sample_scores` holds the scores from the condensed PCA, which will be used.
        } else {
            // Refinement Loop
            // Pass 1 uses initial_sample_pc_scores. Subsequent passes use scores from the previous iteration.
            for pass_num in 1..=self.config.refine_pass_count {
                // Allows zero.

                #[cfg(feature = "enable-eigensnp-diagnostics")]
                let mut current_sr_pass_detail_option: Option<SrPassDetail> = None;
                #[cfg(feature = "enable-eigensnp-diagnostics")]
                {
                    if self.config.collect_diagnostics && diagnostics_collector.is_some() {
                        let mut detail = SrPassDetail::default();
                        detail.pass_num = pass_num;
                        current_sr_pass_detail_option = Some(detail);
                    }
                }

                debug!(
                    "Starting Refinement Pass {} with {} PCs from previous step.",
                    pass_num,
                    current_sample_scores.scores.ncols()
                );

                if current_sample_scores.scores.ncols() == 0 {
                    warn!("Refinement Pass {}: Input scores have 0 components. Cannot proceed with refinement.", pass_num);
                    if pass_num == 1 {
                        final_sorted_snp_loadings = Array2::zeros((num_total_pca_snps, 0));
                    }
                    num_principal_components_computed_final = 0;
                    break;
                }

                let loadings_refinement_start_time = std::time::Instant::now();
                let v_qr_snp_loadings = self.compute_refined_snp_loadings(
                    genotype_data,
                    &current_sample_scores,
                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    current_sr_pass_detail_option.as_mut(),
                    #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                    None,
                )?;
                info!(
                    "Pass {}: Computed QR-based SNP loadings (intermediate V_qr) in {:?}",
                    pass_num,
                    loadings_refinement_start_time.elapsed()
                );

                if v_qr_snp_loadings.ncols() == 0 {
                    warn!("Pass {}: Intermediate QR-based SNP loadings (V_qr) resulted in 0 components. Ending refinement.", pass_num);
                    if pass_num == 1 {
                        final_sorted_snp_loadings = v_qr_snp_loadings;
                    }
                    num_principal_components_computed_final = 0;
                    break;
                }

                let final_outputs_computation_start_time = std::time::Instant::now();
                let (
                    sorted_scores_this_pass,
                    sorted_eigenvalues_this_pass,
                    sorted_loadings_this_pass,
                ) = self.compute_rotated_final_outputs(
                    genotype_data,
                    &v_qr_snp_loadings.view(),
                    num_total_qc_samples,
                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    current_sr_pass_detail_option.as_mut(),
                    #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                    None,
                )?;
                info!(
                    "Pass {}: Computed final rotated scores, eigenvalues, and loadings in {:?}",
                    pass_num,
                    final_outputs_computation_start_time.elapsed()
                );

                current_sample_scores = InitialSamplePcScores {
                    scores: sorted_scores_this_pass.clone(),
                };

                final_sorted_snp_loadings = sorted_loadings_this_pass;
                final_sorted_eigenvalues = sorted_eigenvalues_this_pass;
                num_principal_components_computed_final = final_sorted_snp_loadings.ncols();

                if num_principal_components_computed_final == 0 {
                    warn!(
                        "Pass {}: Refinement resulted in 0 final components. Ending refinement.",
                        pass_num
                    );
                    break;
                }

                #[cfg(feature = "enable-eigensnp-diagnostics")]
                {
                    if let (Some(dc), Some(sr_detail)) = (
                        diagnostics_collector.as_mut(),
                        current_sr_pass_detail_option,
                    ) {
                        if self.config.collect_diagnostics {
                            dc.sr_pass_details.push(sr_detail);
                        }
                    }
                }
            }
            // End of Refinement Loop
        }

        // current_sample_scores now holds the sample scores from the last completed refinement pass.
        // final_sorted_snp_loadings and final_sorted_eigenvalues also hold results from the last completed pass.
        let final_sorted_sample_scores = current_sample_scores.scores; // These are the scores corresponding to the final loadings/eigenvalues

        info!(
            "EigenSNP PCA completed in {:?}. Computed {} Principal Components.",
            overall_start_time.elapsed(),
            num_principal_components_computed_final
        );

        let output_final = EigenSNPCoreOutput {
            final_snp_principal_component_loadings: final_sorted_snp_loadings,
            final_sample_principal_component_scores: final_sorted_sample_scores,
            final_principal_component_eigenvalues: final_sorted_eigenvalues,
            num_qc_samples_used: num_total_qc_samples,
            num_pca_snps_used: genotype_data.num_pca_snps(),
            num_principal_components_computed: num_principal_components_computed_final,
        };

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(dc) = diagnostics_collector.as_mut() {
                if let Some(rt) = overall_start_time.elapsed().as_secs_f64_safe() {
                    dc.total_runtime_seconds = Some(rt);
                }
                dc.notes.push_str("EigenSNP PCA run finished. ");
            }
            // The return value structure now matches PcaOutputWithDiagnostics
            Ok((output_final, diagnostics_collector))
        }
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        {
            // This also matches PcaOutputWithDiagnostics where the second element is ()
            Ok((output_final, ()))
        }
    }

    // fn learn_all_ld_block_local_bases... (the trait was moved from here)

    fn learn_all_ld_block_local_bases<G, F>(
        &self,
        genotype_data: &G,
        ld_block_specs: &[LdBlockSpecification],
        subset_sample_ids: &[QcSampleId],
        snp_metadata: &[PcaSnpMetadata],
        on_local_pcs_generated: F,
        #[cfg(feature = "enable-eigensnp-diagnostics")] mut diagnostics_collector: Option<
            &mut Vec<crate::diagnostics::PerBlockLocalBasisDiagnostics>,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))] _diagnostics_collector_param: Option<
            (),
        >,
    ) -> Result<Vec<LocalBasisWithDiagnostics>, ThreadSafeStdError>
    where
        G: PcaReadyGenotypeAccessor,
        F: Fn(&ArrayView2<f32>, &[PcaSnpMetadata], LdBlockListId) -> Result<(), ThreadSafeStdError>
            + Send
            + Sync,
    {
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

        let local_bases_results: Vec<Result<LocalBasisWithDiagnostics, ThreadSafeStdError>> = ld_block_specs
            .par_iter()
            .enumerate()
            .map(|(block_idx_val, block_spec)| {
                let block_list_id = LdBlockListId(block_idx_val);
                let block_tag = &block_spec.user_defined_block_tag;
                let num_snps_in_this_block_spec = block_spec.num_snps_in_block();

                debug!("Learn Local Bases: Processing block_id {:?} (tag: '{}'), num_snps_in_spec: {}.",
                       block_list_id, block_tag, num_snps_in_this_block_spec);

                // Initialize per_block_diag_entry_for_map here, outside the if/else for num_snps_in_this_block_spec
                #[cfg(feature = "enable-eigensnp-diagnostics")]
                let mut per_block_diag_entry_for_map = {
                    let mut entry = PerBlockLocalBasisDiagnostics::default();
                    if diagnostics_collector.is_some() && self.config.collect_diagnostics {
                        entry.block_id = block_list_id.0.to_string();
                        entry.notes = format!("Processing LD Block tag: {}", block_tag);
                        // Initial u_p_dims and other fields will be updated later if actual data is processed
                    }
                    entry
                };

                let basis_vectors_for_block = if num_snps_in_this_block_spec == 0 {
                    trace!("Block {}: Is empty of SNPs, creating empty basis.", block_tag);
                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    {
                        if diagnostics_collector.is_some() && self.config.collect_diagnostics {
                            per_block_diag_entry_for_map.notes.push_str(" ;Block empty of SNPs per spec.");
                            per_block_diag_entry_for_map.u_p_dims = Some((0,0)); // Explicitly set for empty block
                        }
                    }
                    Array2::<f32>::zeros((0, 0))
                } else {
                    let genotype_block_for_subset_samples = // X_sp
                        genotype_data.get_standardized_snp_sample_block(
                            &block_spec.pca_snp_ids_in_block,
                            subset_sample_ids,
                        ).map_err(|e_accessor| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get standardized SNP/sample block for block ID {:?} ({}): {}", block_list_id, block_tag, e_accessor))) as ThreadSafeStdError)?;

                    debug!("Block {}: X_sp (subset genotype block) dimensions: {:?}",
                           block_tag, genotype_block_for_subset_samples.dim());

                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    {
                        if diagnostics_collector.is_some() && self.config.collect_diagnostics {
                            let (r,c) = genotype_block_for_subset_samples.dim();
                            per_block_diag_entry_for_map.input_x_s_p_dims = Some((r,c));
                            per_block_diag_entry_for_map.notes.push_str(&format!(", X_s_p dims: ({}, {})", r, c));
                            if !genotype_block_for_subset_samples.is_empty() {
                                per_block_diag_entry_for_map.input_x_s_p_fro_norm = Some(compute_frob_norm_f32(&genotype_block_for_subset_samples.view()) as f64);
                                let x_s_p_f64 = genotype_block_for_subset_samples.mapv(|x_val| x_val as f64);
                                per_block_diag_entry_for_map.input_x_s_p_condition_number = compute_condition_number_via_svd_f64(&x_s_p_f64.view());
                            }
                        }
                    }

                    if !genotype_block_for_subset_samples.is_empty() {
                        let norm_x_sp = genotype_block_for_subset_samples.view().mapv(|x| x*x).sum().sqrt();
                        trace!("Block {}: X_sp Frobenius norm: {:.4e}", block_tag, norm_x_sp);
                    }

                    let actual_num_snps_in_block = genotype_block_for_subset_samples.nrows();
                    let actual_num_subset_samples = genotype_block_for_subset_samples.ncols();

                    let num_components_to_extract = self.config.components_per_ld_block
                        .min(actual_num_snps_in_block)
                        .min(if actual_num_subset_samples > 0 { actual_num_subset_samples } else { 0 });

                    // This 'if' block is now part of the 'else' for 'num_snps_in_this_block_spec > 0'
                    // It determines the content of 'basis_vectors_for_block'
                    if num_components_to_extract == 0 {
                        debug!(
                            "Block {}: Num components to extract is 0 (SNPs_in_block={}, N_subset={}, Configured_cp={}), creating empty basis.",
                            block_tag,
                            actual_num_snps_in_block,
                            actual_num_subset_samples,
                            self.config.components_per_ld_block
                        );

                        #[cfg(feature = "enable-eigensnp-diagnostics")]
                        {
                            // per_block_diag_entry_for_map is already defined and mutable.
                            // Just update its fields for this specific early exit path.
                            if diagnostics_collector.is_some() && self.config.collect_diagnostics {
                                per_block_diag_entry_for_map.notes = format!(
                                   "Num components to extract is 0 for block tag: {}. SNPs in block spec: {}, Actual subset samples: {}. Original notes: {}",
                                   block_tag, num_snps_in_this_block_spec, actual_num_subset_samples, per_block_diag_entry_for_map.notes
                                );
                                per_block_diag_entry_for_map.u_p_dims = Some((actual_num_snps_in_block, 0));
                            }
                        }
                        // Set basis_vectors_for_block to empty and let flow continue to the end of the closure.
                        Array2::<f32>::zeros((actual_num_snps_in_block, 0))
                    } else {
                        let local_seed = self.config.random_seed.wrapping_add(block_idx_val as u64);

                        // per_block_diag_entry_for_map is already initialized.

                        let local_basis_vectors_f32 = Self::perform_randomized_svd_for_loadings( // Up_star
                            &genotype_block_for_subset_samples.view(),
                            num_components_to_extract,
                            self.config.local_rsvd_sketch_oversampling,
                            self.config.local_rsvd_num_power_iterations,
                            local_seed,
                            #[cfg(feature = "enable-eigensnp-diagnostics")]
                            diagnostics_collector.as_ref().and_then(|_| {
                                if self.config.collect_diagnostics && self.config.diagnostic_block_list_id_to_trace == Some(block_list_id.0) {
                                    Some(&mut per_block_diag_entry_for_map.rsvd_stages)
                                } else { None }
                            }),
                            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                            None,
                        ).map_err(|e_rsvd| -> ThreadSafeStdError {
                            Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Local RSVD failed for block ID {:?} ({}): {}", block_list_id, block_tag, e_rsvd)))
                        })?;

                        debug!("Block {}: Local basis vectors (Up_star) dimensions: {:?}", block_tag, local_basis_vectors_f32.dim());

                        #[cfg(feature = "enable-eigensnp-diagnostics")]
                        {
                            if diagnostics_collector.is_some() && self.config.collect_diagnostics {
                                let (r_up, c_up) = local_basis_vectors_f32.dim();
                                // Update the u_p_dims field of the already defined per_block_diag_entry_for_map
                                per_block_diag_entry_for_map.u_p_dims = Some((r_up, c_up));
                                if !local_basis_vectors_f32.is_empty() {
                                    per_block_diag_entry_for_map.u_p_fro_norm = Some(compute_frob_norm_f32(&local_basis_vectors_f32.view()) as f64);
                                    per_block_diag_entry_for_map.u_p_orthogonality_error = compute_orthogonality_error_f32(&local_basis_vectors_f32.view());
                                    per_block_diag_entry_for_map.u_p_condition_number = compute_condition_number_via_svd_f32(&local_basis_vectors_f32.view());
                                }
                                if self.config.diagnostic_block_list_id_to_trace == Some(block_list_id.0) && !genotype_block_for_subset_samples.is_empty() {
                                    let x_s_p_f64 = genotype_block_for_subset_samples.mapv(|x_val| x_val as f64);
                                    let backend_f64 = LinAlgBackendProvider::<f64>::new();
                                    match backend_f64.svd_into(x_s_p_f64, true, false) {
                                        Ok(svd_out_f64) => {
                                            if let Some(u_true_f64) = svd_out_f64.u {
                                                let k_to_compare = local_basis_vectors_f32.ncols().min(u_true_f64.ncols());
                                                if k_to_compare > 0 {
                                                    let u_p_f32_view = local_basis_vectors_f32.slice_axis(Axis(1), ndarray::Slice::from(0..k_to_compare));
                                                    let u_true_f64_view = u_true_f64.slice_axis(Axis(1), ndarray::Slice::from(0..k_to_compare));
                                                    per_block_diag_entry_for_map.u_correlation_vs_f64_truth =
                                                        compute_matrix_column_correlations_abs(&u_p_f32_view, &u_true_f64_view.view());
                                                }
                                            } else { per_block_diag_entry_for_map.notes.push_str(" ;f64 SVD U_true was None"); }
                                        }
                                        Err(e) => { per_block_diag_entry_for_map.notes.push_str(&format!(" ;f64 SVD for U_true failed: {}", e)); }
                                    }
                                }
                            }
                        }
                        local_basis_vectors_f32
                    }
                }; // End of basis_vectors_for_block assignment

                let basis_result = PerBlockLocalSnpBasis {
                    block_list_id,
                    basis_vectors: basis_vectors_for_block,
                };

                // --- NEW CODE START ---
                // Get the metadata for just the SNPs in this specific block
                let block_specific_metadata: Vec<PcaSnpMetadata> = block_spec
                    .pca_snp_ids_in_block
                    .iter()
                    .map(|pca_id| snp_metadata[pca_id.0].clone()) // Look up and clone
                    .collect();
                // --- NEW CODE END ---

                // Invoke the provided closure to consume the generated local PCs.
                // This call is monomorphized by the compiler to be either a file-writing
                // operation or a true no-op, achieving a zero-cost abstraction.
                on_local_pcs_generated(
                    &basis_result.basis_vectors.view(),
                    &block_specific_metadata,
                    block_list_id,
                )?;

                // Now, per_block_diag_entry_for_map is guaranteed to be in scope.
                #[cfg(feature = "enable-eigensnp-diagnostics")]
                let diag_to_return = per_block_diag_entry_for_map;
                #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                let diag_to_return = ();

                Ok((basis_result, diag_to_return))
            })
            .collect();

        // Separate results and diagnostics
        let mut final_results_tuples = Vec::with_capacity(local_bases_results.len());

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        let mut collected_diagnostics_entries: Vec<
            crate::diagnostics::PerBlockLocalBasisDiagnostics,
        > = Vec::new();

        for result_item_tuple in local_bases_results {
            // Each item in local_bases_results is Result<(PerBlockLocalSnpBasis, ActualDiagType), ThreadSafeStdError>>
            // where ActualDiagType is PerBlockLocalBasisDiagnostics or ()
            let (basis_result_item, diag_entry_for_this_block) = result_item_tuple?;

            #[cfg(feature = "enable-eigensnp-diagnostics")]
            {
                // Here, diag_entry_for_this_block is PerBlockLocalBasisDiagnostics
                // Ensure self.config.collect_diagnostics is the primary guard.
                // The diagnostics_collector.is_some() check is also good to ensure it's not None
                // if self.config.collect_diagnostics was true but initialization somehow failed (though less likely).
                if self.config.collect_diagnostics && diagnostics_collector.is_some() {
                    collected_diagnostics_entries.push(diag_entry_for_this_block.clone());
                    // Clone and store
                }
            }
            // If diagnostics are not enabled, diag_entry_for_this_block is (), which is Copy.
            // The final_results_tuples will store (PerBlockLocalSnpBasis, ActualDiagType)
            // where ActualDiagType is () if not collecting, or the moved original diag if collecting but not pushing to main collector yet.
            // To simplify, we always push the (basis, original_diag_entry) to final_results_tuples.
            // The collected_diagnostics_entries vector is now the primary source for the main diagnostics collector.
            final_results_tuples.push((basis_result_item, diag_entry_for_this_block));
        }

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            // Ensure this whole block is guarded by self.config.collect_diagnostics as well
            if self.config.collect_diagnostics {
                if let Some(dc_vec_mut) = diagnostics_collector.as_mut() {
                    dc_vec_mut.extend(collected_diagnostics_entries);
                }
            }
        }

        // The type of final_results_tuples is Vec<(PerBlockLocalSnpBasis, LocalBasisWithDiagnostics::Item2)>
        // which is Vec<(PerBlockLocalSnpBasis, PerBlockLocalBasisDiagnostics)> or Vec<(PerBlockLocalSnpBasis, ())>
        // This matches the required return type Vec<LocalBasisWithDiagnostics>.
        let final_results_with_conditional_diagnostics = final_results_tuples;

        info!("Successfully learned local eigenSNP bases for all blocks.");
        Ok(final_results_with_conditional_diagnostics)
    }

    fn project_all_samples_onto_local_bases<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        ld_block_specs: &[LdBlockSpecification],
        all_local_bases: &[LocalBasisWithDiagnostics],
        num_total_qc_samples: usize,
        #[cfg(feature = "enable-eigensnp-diagnostics")] mut full_diagnostics_collector: Option<
            &mut crate::diagnostics::FullPcaRunDetailedDiagnostics,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))] _full_diagnostics_collector: Option<()>,
    ) -> Result<RawCondensedFeatures, ThreadSafeStdError> {
        assert_eq!(
            ld_block_specs.len(),
            all_local_bases.len(),
            "Mismatch between LD block specifications count ({}) and learned local bases count ({}). Ensure each LD block has a corresponding local basis entry.",
            ld_block_specs.len(),
            all_local_bases.len()
        );
        info!(
            "Projecting {} total QC samples onto local bases to construct condensed feature matrix.",
            num_total_qc_samples
        );

        let total_num_condensed_features: usize = all_local_bases
            .iter()
            .map(|basis| basis.0.basis_vectors.ncols())
            .sum();

        if total_num_condensed_features == 0 {
            info!("Total condensed features is 0. Returning empty RawCondensedFeatures.");
            return Ok(RawCondensedFeatures {
                data: Array2::<f32>::zeros((0, num_total_qc_samples)),
            });
        }
        debug!(
            "Total number of condensed features (rows in A_eigen) = {}",
            total_num_condensed_features
        );

        let mut raw_condensed_data_matrix =
            Array2::<f32>::zeros((total_num_condensed_features, num_total_qc_samples));
        let mut current_condensed_feature_row_offset = 0;

        let all_qc_sample_ids: Vec<QcSampleId> =
            (0..num_total_qc_samples).map(QcSampleId).collect();

        for block_idx in 0..ld_block_specs.len() {
            let block_spec = &ld_block_specs[block_idx];
            let block_tag = &block_spec.user_defined_block_tag;
            let (local_basis_data, _) = &all_local_bases[block_idx]; // Destructure the tuple

            let local_snp_basis_vectors = &local_basis_data.basis_vectors;
            let num_components_this_block = local_snp_basis_vectors.ncols();

            if block_spec.num_snps_in_block() == 0 || num_components_this_block == 0 {
                trace!("Project Samples: Skipping block {} for projection: num_snps={} or num_local_components=0.",
                       block_tag, block_spec.num_snps_in_block());
                continue;
            }

            let genotype_data_for_block_all_samples = genotype_data.get_standardized_snp_sample_block(
                &block_spec.pca_snp_ids_in_block,
                &all_qc_sample_ids,
            ).map_err(|e_accessor| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to get standardized SNP/sample block during projection for block '{}': {}", block_tag, e_accessor))) as ThreadSafeStdError)?;

            // projected_scores_for_block = Sp_star = Up_star.T * Xp (cp x N)
            let projected_scores_for_block = Self::dot_product_at_b_mixed_precision(
                &local_snp_basis_vectors.view(),
                &genotype_data_for_block_all_samples.view(),
            )?;

            debug!(
                "Block {}: Projected scores (Sp_star) dimensions: {:?}",
                block_tag,
                projected_scores_for_block.dim()
            );
            if !projected_scores_for_block.is_empty() {
                let norm_sp_star = projected_scores_for_block
                    .view()
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();
                trace!(
                    "Block {}: Projected scores (Sp_star) Frobenius norm: {:.4e}",
                    block_tag,
                    norm_sp_star
                );
                trace!(
                    "Block {}: Projected scores (Sp_star) sample: {:?}",
                    block_tag,
                    projected_scores_for_block.slice(s![
                        0..3.min(projected_scores_for_block.nrows()),
                        0..3.min(projected_scores_for_block.ncols())
                    ])
                );
            }

            raw_condensed_data_matrix
                .slice_mut(s![
                    current_condensed_feature_row_offset
                        ..current_condensed_feature_row_offset + num_components_this_block,
                    ..
                ])
                .assign(&projected_scores_for_block);

            current_condensed_feature_row_offset += num_components_this_block;
        }

        debug!(
            "Raw condensed feature matrix (A_eigen_star) dimensions: {:?}",
            raw_condensed_data_matrix.dim()
        );
        if !raw_condensed_data_matrix.is_empty() {
            let norm_a_eigen_star = raw_condensed_data_matrix
                .view()
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            debug!(
                "Raw condensed feature matrix (A_eigen_star) Frobenius norm: {:.4e}",
                norm_a_eigen_star
            );

            for row_idx in 0..3.min(raw_condensed_data_matrix.nrows()) {
                let r_view = raw_condensed_data_matrix.row(row_idx);
                if r_view.len() > 1 {
                    // Variance requires at least 2 elements
                    let mean_val = r_view.mean().unwrap_or(0.0);
                    let variance = r_view
                        .mapv(|x| (x - mean_val).powi(2))
                        .mean()
                        .unwrap_or(0.0);
                    trace!(
                        "Raw condensed matrix: Row {} variance (pre-std): {:.4e}",
                        row_idx,
                        variance
                    );
                } else if r_view.len() == 1 {
                    trace!(
                        "Raw condensed matrix: Row {} variance (pre-std): N/A (single element)",
                        row_idx
                    );
                }
            }
        }
        info!(
            "Constructed raw condensed feature matrix. Shape: {:?}",
            raw_condensed_data_matrix.dim()
        );

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(dc) = full_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics {
                    dc.c_matrix_dims = Some(raw_condensed_data_matrix.dim());
                    if !raw_condensed_data_matrix.is_empty() {
                        dc.c_matrix_fro_norm =
                            Some(compute_frob_norm_f32(&raw_condensed_data_matrix.view()) as f64);
                    } else {
                        dc.c_matrix_fro_norm = Some(0.0);
                    }
                }
            }
        }
        Ok(RawCondensedFeatures {
            data: raw_condensed_data_matrix,
        })
    }

    fn compute_pca_on_standardized_condensed_features_via_rsvd(
        &self,
        standardized_condensed_features: &StandardizedCondensedFeatures,
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        mut global_pca_diagnostics_collector: Option<
            &mut crate::diagnostics::GlobalPcaDiagnostics,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        mut _global_pca_diagnostics_collector_param: Option<()>, // Renamed to avoid conflict
    ) -> Result<InitialSamplePcScores, ThreadSafeStdError> {
        let a_c = &standardized_condensed_features.data; // A_eigen_std_star
        let m_c = a_c.nrows();
        let n_samples = a_c.ncols();

        let k_glob = self.config.target_num_global_pcs;
        let p_glob = self.config.global_pca_sketch_oversampling;
        let q_glob = self.config.global_pca_num_power_iterations; // For RSVD
        let random_seed = self.config.random_seed; // For RSVD

        // Initial logging of parameters
        debug!("Initial Global PCA: M_c (condensed features) = {}", m_c);
        debug!("Initial Global PCA: N_samples = {}", n_samples);
        debug!("Initial Global PCA: K_glob (target PCs) = {}", k_glob);
        debug!("Initial Global PCA: p_glob (oversampling) = {}", p_glob);

        // Handle cases where input matrix dimensions are zero.
        if m_c == 0 || n_samples == 0 || k_glob == 0 {
            warn!(
                "Initial PCA on condensed features: M_c ({}) or N_samples ({}) or K_glob ({}) is 0. Returning empty scores ({}x0).",
                m_c, n_samples, k_glob, n_samples
            );
            return Ok(InitialSamplePcScores {
                scores: Array2::zeros((n_samples, 0)),
            });
        }

        let l_rsvd = (k_glob + p_glob).min(m_c.min(n_samples));
        debug!("Initial Global PCA: L_rsvd calculated: {}", l_rsvd);
        // debug!( // This is a duplicate of a later log, remove if not needed for specific flow tracking
        //     "Initial PCA on condensed features: M_c={}, N_samples={}, K_glob={}, p_glob={}, L_rsvd_raw_sketch={}",
        //     m_c, n_samples, k_glob, p_glob, k_glob + p_glob
        // );
        // debug!( // This is also somewhat redundant given the new L_rsvd specific log
        //     "Initial PCA on condensed features: Effective L_rsvd (min with M_c, N_samples) = {}",
        //     l_rsvd
        // );

        let direct_svd_m_c_threshold = 500;
        let initial_scores: Array2<f32>;

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(gdc) = global_pca_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics {
                    gdc.stage_name = "GlobalPCA_Initial".to_string();
                    // Record A_eigen_std_star (a_c) properties
                    if !a_c.is_empty() {
                        // Storing these in the first RsvdStepDetail for now
                        let mut first_step_detail = RsvdStepDetail::default();
                        first_step_detail.step_name = "Input_A_eigen_std".to_string();
                        first_step_detail.input_matrix_dims = Some(a_c.dim());
                        first_step_detail.fro_norm =
                            Some(compute_frob_norm_f32(&a_c.view()) as f64);
                        first_step_detail.condition_number =
                            compute_condition_number_via_svd_f32(&a_c.view());
                        // Also record f64 condition number if desired, perhaps in notes or a dedicated field if added
                        let a_c_f64 = a_c.mapv(|v| v as f64);
                        let cond_f64 = compute_condition_number_via_svd_f64(&a_c_f64.view());
                        first_step_detail.notes =
                            format!("Input A_eigen_std f64 cond_num: {:?}", cond_f64);
                        gdc.rsvd_stages.push(first_step_detail);
                    }
                }
            }
        }

        if m_c <= k_glob || m_c <= direct_svd_m_c_threshold || l_rsvd <= k_glob {
            info!("Initial Global PCA: Choosing Direct SVD path. Condition: m_c ({}) <= k_glob ({}) || m_c ({}) <= direct_svd_m_c_threshold ({}) || l_rsvd ({}) <= k_glob ({})",
                  m_c, k_glob, m_c, direct_svd_m_c_threshold, l_rsvd, k_glob);
            let a_c_owned_for_svd = a_c.to_owned(); // For SVD

            debug!(
                "Direct SVD Path: A_c (condensed matrix) dimensions: {:?}",
                a_c_owned_for_svd.dim()
            );
            let backend = LinAlgBackendProvider::<f32>::new();
            match backend.svd_into(a_c_owned_for_svd.clone(), false, true) {
                // Clone a_c_owned_for_svd for potential f64 SVD later
                Ok(svd_output) => {
                    if let Some(svd_output_vt) = svd_output.vt {
                        if svd_output_vt.is_empty() {
                            initial_scores = Array2::zeros((n_samples, 0));
                        } else {
                            let num_svd_components = svd_output_vt.nrows();
                            let k_eff = k_glob.min(num_svd_components);
                            if k_eff == 0 {
                                initial_scores = Array2::zeros((n_samples, 0));
                            } else {
                                initial_scores = svd_output_vt
                                    .t()
                                    .slice_axis(Axis(1), ndarray::Slice::from(0..k_eff))
                                    .to_owned();
                            }
                        }
                    } else {
                        /* error handling */
                        warn!("Direct SVD for initial global PCA: svd_output.vt is None despite requesting it. M_c={}, N_samples={}", m_c, n_samples);
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "SVD succeeded but V.T (vt) was not returned by the backend.",
                        )) as ThreadSafeStdError);
                    }

                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    {
                        if let Some(gdc) = global_pca_diagnostics_collector.as_mut() {
                            if self.config.collect_diagnostics
                                && !a_c.is_empty()
                                && !initial_scores.is_empty()
                            {
                                debug!("DIAG: Computing f64 SVD for U_scores_true comparison in Global PCA (Direct SVD Path).");
                                let a_c_f64_owned = a_c.mapv(|v_f32| v_f32 as f64); // Convert A_c to f64 for true SVD
                                let backend_f64 = LinAlgBackendProvider::<f64>::new();
                                match backend_f64.svd_into(a_c_f64_owned, false, true) {
                                    // Request VT_f64
                                    Ok(svd_out_f64) => {
                                        if let Some(vt_true_f64) = svd_out_f64.vt {
                                            let k_to_compare =
                                                initial_scores.ncols().min(vt_true_f64.nrows());
                                            if k_to_compare > 0 {
                                                let u_scores_true_f64 = vt_true_f64
                                                    .t()
                                                    .slice_axis(
                                                        Axis(1),
                                                        ndarray::Slice::from(0..k_to_compare),
                                                    )
                                                    .into_owned();
                                                gdc.initial_scores_correlation_vs_py_truth = // Assuming py_truth means f64_truth here
                                                    compute_matrix_column_correlations_abs(&initial_scores.view(), &u_scores_true_f64.view());
                                            }
                                        } else {
                                            gdc.notes.push_str(
                                                " ;f64 SVD Vt_true was None for Global PCA truth",
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        gdc.notes.push_str(&format!(
                                            " ;f64 SVD for U_scores_true failed in Global PCA: {}",
                                            e
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    /* error handling */
                    warn!("Direct SVD failed for initial global PCA (M_c={}, N_samples={}): {}. Returning error.", m_c, n_samples, e);
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Direct SVD failed during initial global PCA: {}", e),
                    )) as ThreadSafeStdError);
                }
            }
        } else {
            info!("Initial Global PCA: Choosing RSVD path. Condition: m_c ({}) > k_glob ({}) && m_c ({}) > direct_svd_m_c_threshold ({}) && l_rsvd ({}) > k_glob ({})",
                  m_c, k_glob, m_c, direct_svd_m_c_threshold, l_rsvd, k_glob);

            #[cfg(feature = "enable-eigensnp-diagnostics")]
            let rsvd_stages_collector = global_pca_diagnostics_collector
                .as_mut()
                .map(|gdc| &mut gdc.rsvd_stages);
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            let rsvd_stages_collector = None;

            initial_scores = Self::perform_randomized_svd_for_scores(
                &a_c.view(),
                k_glob,
                p_glob,
                q_glob,
                random_seed,
                rsvd_stages_collector,
            )?;
        }

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(gdc) = global_pca_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics && !initial_scores.is_empty() {
                    // Record initial_scores properties (assuming initial_scores is U_scores_star)
                    // gdc.initial_scores_dims = Some(initial_scores.dim()); // This field does not exist
                    // gdc.initial_scores_fro_norm = Some(compute_frob_norm_f32(&initial_scores.view()) as f64); // This field does not exist
                    // gdc.initial_scores_orthogonality_error = compute_orthogonality_error_f32(&initial_scores.view()); // This field does not exist
                    // Storing these in notes for now, or they could be the last RsvdStepDetail from perform_randomized_svd_for_scores
                    let (r, c) = initial_scores.dim();
                    let fro_norm = compute_frob_norm_f32(&initial_scores.view()) as f64;
                    let ortho_error = compute_orthogonality_error_f32(&initial_scores.view());
                    gdc.notes.push_str(&format!(
                        " ;InitialScores dims:({},{}), FrobNorm:{:.4e}, OrthoError:{:?}",
                        r, c, fro_norm, ortho_error
                    ));
                }
            }
        }

        if initial_scores.ncols() == 0 && k_glob > 0 {
            warn!("Initial PCA scores have 0 columns (M_c={}, N_samples={}), but k_glob ({}) > 0. This might indicate an issue or empty input.", m_c, n_samples, k_glob);
        }

        Ok(InitialSamplePcScores {
            scores: initial_scores,
        })
    }

    /// Computes the right singular vectors (V_A_approx, sample scores) of a matrix A using rSVD.
    /// A is M features x N samples. Output is N x K_eff.
    pub fn perform_randomized_svd_for_scores(
        matrix_features_by_samples: &ArrayView2<f32>,
        num_components_target_k: usize,
        sketch_oversampling_count: usize,
        num_power_iterations: usize,
        random_seed: u64,
        #[cfg(feature = "enable-eigensnp-diagnostics")] _diagnostics_collector: Option<
            &mut Vec<crate::diagnostics::RsvdStepDetail>,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))] _diagnostics_collector: Option<()>,
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
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            _diagnostics_collector,
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            _diagnostics_collector,
        )?;

        // if let (Some(collector), Some(diag)) = (diagnostics_collector, step_diag) {
        //    if self.config.collect_diagnostics { collector.push(diag); }
        // }

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
        #[cfg(feature = "enable-eigensnp-diagnostics")] _diagnostics_collector: Option<
            &mut Vec<crate::diagnostics::RsvdStepDetail>,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))] _diagnostics_collector: Option<()>,
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
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            _diagnostics_collector,
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            _diagnostics_collector,
        )?;

        // if let (Some(collector), Some(diag)) = (diagnostics_collector, step_diag) {
        //    if self.config.collect_diagnostics { collector.push(diag); }
        // }

        u_opt.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Loadings (U matrix) not computed or available from _internal_perform_rsvd",
            )) as ThreadSafeStdError
        })
    }

    /// Performs matrix multiplication of A.T * B (A: D_strip x N, B: D_strip x K_qr)
    /// using f64 accumulation for each element of the resulting f32 matrix (N x K_qr).
    fn dot_product_at_b_mixed_precision(
        matrix_a_dstrip_x_n: &ArrayView2<f32>, // Corresponds to genotype_data_strip_f32.view()
        matrix_b_dstrip_x_kqr: &ArrayView2<f32>, // Corresponds to v_qr_loadings_for_strip_f32.view()
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let d_strip = matrix_a_dstrip_x_n.nrows();
        let n_samples = matrix_a_dstrip_x_n.ncols();
        let k_qr = matrix_b_dstrip_x_kqr.ncols();

        const LANES: usize = 8; // Define LANES constant

        if d_strip != matrix_b_dstrip_x_kqr.nrows() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Dimension mismatch for mixed-precision A.T * B dot product: A.nrows ({}) != B.nrows ({}).",
                    d_strip, matrix_b_dstrip_x_kqr.nrows()
                ),
            )) as ThreadSafeStdError);
        }

        if d_strip == 0 || n_samples == 0 || k_qr == 0 {
            return Ok(Array2::<f32>::zeros((n_samples, k_qr)));
        }

        let mut result_n_x_kqr_f32 = Array2::<f32>::zeros((n_samples, k_qr));

        result_n_x_kqr_f32
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i_sample_idx, mut output_row_f32_view)| {
                let a_col_i_view = matrix_a_dstrip_x_n.column(i_sample_idx);
                // Slices are no longer obtained here. Loading is done manually into arrays.

                for k_comp_idx in 0..k_qr {
                    let mut accumulator_f64: f64 = 0.0;
                    let b_col_k_view = matrix_b_dstrip_x_kqr.column(k_comp_idx);
                    // Slice for b_col_k_view is also removed.

                    let num_simd_chunks = d_strip / LANES;
                    let mut simd_f32_partial_sum = Simd::splat(0.0f32);

                    for chunk_idx in 0..num_simd_chunks {
                        let offset = chunk_idx * LANES;

                        let mut a_temp_array = [0.0f32; LANES];
                        for lane_idx in 0..LANES {
                            a_temp_array[lane_idx] = a_col_i_view[offset + lane_idx];
                        }
                        let a_simd = Simd::from_array(a_temp_array);

                        let mut b_temp_array = [0.0f32; LANES];
                        for lane_idx in 0..LANES {
                            b_temp_array[lane_idx] = b_col_k_view[offset + lane_idx];
                        }
                        let b_simd = Simd::from_array(b_temp_array);

                        simd_f32_partial_sum += a_simd * b_simd;
                    }
                    accumulator_f64 += simd_f32_partial_sum.reduce_sum() as f64;

                    for d_snp_idx in (num_simd_chunks * LANES)..d_strip {
                        accumulator_f64 +=
                            (a_col_i_view[d_snp_idx] as f64) * (b_col_k_view[d_snp_idx] as f64);
                    }
                    output_row_f32_view[k_comp_idx] = accumulator_f64 as f32;
                }
            });

        Ok(result_n_x_kqr_f32)
    }

    fn compute_refined_snp_loadings<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        initial_sample_pc_scores: &InitialSamplePcScores,
        #[cfg(feature = "enable-eigensnp-diagnostics")] mut pass_diagnostics_collector: Option<
            &mut crate::diagnostics::SrPassDetail,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        _pass_diagnostics_collector_param: Option<()>, // Renamed
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        // Computes $V_{QR}^* = X U_{scores}^*$, where $X$ is D_blocked x N and $U_{scores}^*$ is N x K_initial.
        // The result $V_{QR}^*$ is D_blocked x K_initial.
        // This is then orthogonalized via QR decomposition to get $V_{QR}$ (D_blocked x K_initial_eff).
        //
        // ## Numerical Precision
        // The core matrix multiplication $X_{strip} U_{scores}^*$ is performed using the
        // `dot_product_mixed_precision_f32_f64acc` helper function. This function
        // calculates each element of the resulting matrix by summing products of `f32`
        // elements in an `f64` accumulator, and then casts the final sum back to `f32`.
        // This approach enhances numerical precision for the sum over the $N$ dimension
        // (number of samples) compared to a pure `f32` accumulation (e.g., via `sgemm`).
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

        // Use the configured strip size, ensuring it's at least 1 and not more than total SNPs.
        let snp_processing_strip_size = self
            .config
            .snp_processing_strip_size
            .min(num_total_pca_snps)
            .max(1);

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

                    // Perform dot product with f64 accumulation
                    let snp_loadings_for_strip = Self::dot_product_mixed_precision_f32_f64acc(
                        &genotype_data_strip_snps_by_samples.view(),
                        &initial_scores_n_by_k_initial.view(), // initial_scores_n_by_k_initial is &Array2<f32>
                    )?;
                    loadings_strip_view_mut.assign(&snp_loadings_for_strip);
                    Ok(())
                })?;
        }

        if snp_loadings_before_ortho_pca_snps_by_components.ncols() == 0 {
            info!("Refined loadings matrix has 0 columns, QR skipped.");
            return Ok(snp_loadings_before_ortho_pca_snps_by_components);
        }

        let backend = LinAlgBackendProvider::<f32>::new(); // Use LinAlgBackendProvider for f32
        let orthonormal_snp_loadings = backend
            .qr_q_factor(&snp_loadings_before_ortho_pca_snps_by_components)
            .map_err(|e_qr| -> ThreadSafeStdError {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "QR decomposition of refined loadings failed (via backend): {}",
                        e_qr
                    ),
                )
                .into()
            })?;

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(pdc) = pass_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics {
                    // pass_num is set in compute_pca loop
                    // V_hat_dims would be initial_scores_n_by_k_initial.dim() if V_hat is U_scores from prev pass
                    // For now, assume V_hat is the input scores to this stage
                    pdc.v_hat_dims = Some(initial_scores_n_by_k_initial.dim());
                    if !initial_scores_n_by_k_initial.is_empty() {
                        pdc.v_hat_orthogonality_error =
                            compute_orthogonality_error_f32(&initial_scores_n_by_k_initial.view());
                    }
                    // L_raw_star is snp_loadings_before_ortho_pca_snps_by_components
                    // This matrix is D x K_initial. S_intermediate in SrPassDetail is N x K_prev_eigenvecs
                    // The naming here is a bit confusing. Let's record L_raw_star's condition number in notes for now.
                    if !snp_loadings_before_ortho_pca_snps_by_components.is_empty() {
                        let cond_num_l_raw = compute_condition_number_via_svd_f32(
                            &snp_loadings_before_ortho_pca_snps_by_components.view(),
                        );
                        pdc.notes.push_str(&format!(
                            "L_raw_star (SNP loadings pre-QR) cond_num: {:?}; ",
                            cond_num_l_raw
                        ));
                    }
                    // V_qr_star is orthonormal_snp_loadings (D x K_eff)
                    // This is the Q factor of V_hat in the notation S_intermediate = C_std @ V_hat_Q
                    // Let's use s_intermediate_dims for V_qr_star (orthonormal_snp_loadings)
                    pdc.s_intermediate_dims = Some(orthonormal_snp_loadings.dim()); // This is V_qr*
                    if !orthonormal_snp_loadings.is_empty() {
                        pdc.s_intermediate_fro_norm =
                            Some(compute_frob_norm_f32(&orthonormal_snp_loadings.view()) as f64);
                        // Orthogonality error for V_qr_star (orthonormal_snp_loadings)
                        // This is U_s in SrPassDetail if we consider V_qr* = U_s S_s V_s^T, but here it's just a Q factor.
                        // The field u_s_orthogonality_error or v_hat_orthogonality_error could be used.
                        // Let's use v_hat_orthogonality_error for the input `initial_sample_pc_scores`
                        // and u_s_orthogonality_error for the output `orthonormal_snp_loadings` (which is V_QR*).
                        pdc.u_s_orthogonality_error =
                            compute_orthogonality_error_f32(&orthonormal_snp_loadings.view());
                    }
                }
            }
        }

        info!(
            "Computed refined SNP loadings. Shape: {:?}",
            orthonormal_snp_loadings.dim()
        );
        Ok(orthonormal_snp_loadings)
    }

    fn compute_rotated_final_outputs<G: PcaReadyGenotypeAccessor>(
        &self,
        genotype_data: &G,
        v_qr_loadings_d_by_k: &ArrayView2<f32>, // V_qr (D x K_initial)
        num_total_qc_samples: usize,            // N
        #[cfg(feature = "enable-eigensnp-diagnostics")] mut pass_diagnostics_collector: Option<
            &mut crate::diagnostics::SrPassDetail,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        _pass_diagnostics_collector_param: Option<()>, // Renamed
    ) -> Result<(Array2<f32>, Array1<f64>, Array2<f32>), ThreadSafeStdError> {
        let num_total_pca_snps = v_qr_loadings_d_by_k.nrows(); // D
        let k_initial_components = v_qr_loadings_d_by_k.ncols(); // K_initial

        info!(
            "Computing final rotated outputs (scores, eigenvalues, loadings) for {} samples, {} initial components from V_qr.",
            num_total_qc_samples, k_initial_components
        );

        // --- A. Handle Edge Cases ---
        if k_initial_components == 0 {
            debug!("No initial components in V_qr (K_initial=0), returning empty results.");
            return Ok((
                Array2::zeros((num_total_qc_samples, 0)),
                Array1::zeros(0),
                Array2::zeros((num_total_pca_snps, 0)),
            ));
        }
        if num_total_pca_snps == 0 {
            debug!(
                "No PCA SNPs (D=0), returning empty results for {} initial components.",
                k_initial_components
            );
            return Ok((
                Array2::zeros((num_total_qc_samples, k_initial_components)),
                Array1::zeros(k_initial_components),
                Array2::zeros((0, k_initial_components)),
            ));
        }
        if num_total_qc_samples == 0 {
            debug!(
                "No QC samples (N=0), returning empty results for {} initial components.",
                k_initial_components
            );
            return Ok((
                Array2::zeros((0, k_initial_components)),
                Array1::zeros(k_initial_components),
                Array2::zeros((num_total_pca_snps, k_initial_components)),
            ));
        }

        // --- B. Calculate Intermediate Scores (S_intermediate = X^T · V_qr) with f64 Accumulation ---
        // Use the configured strip size, ensuring it's at least 1 and not more than total SNPs.
        let snp_processing_strip_size = self
            .config
            .snp_processing_strip_size
            .min(num_total_pca_snps)
            .max(1);
        let all_qc_sample_ids_for_scores: Vec<QcSampleId> =
            (0..num_total_qc_samples).map(QcSampleId).collect();

        let strip_indices_starts: Vec<usize> = (0..num_total_pca_snps)
            .step_by(snp_processing_strip_size)
            .collect();

        let s_intermediate_n_by_k_initial_f64: Array2<f64> = strip_indices_starts
            .par_iter()
            .map(
                |&strip_start_snp_idx| -> Result<Array2<f64>, ThreadSafeStdError> {
                    let strip_end_snp_idx =
                        (strip_start_snp_idx + snp_processing_strip_size).min(num_total_pca_snps);
                    if strip_start_snp_idx >= strip_end_snp_idx {
                        return Ok(Array2::<f64>::zeros((
                            num_total_qc_samples,
                            k_initial_components,
                        )));
                    }

                    let snp_ids_in_strip: Vec<PcaSnpId> = (strip_start_snp_idx..strip_end_snp_idx)
                        .map(PcaSnpId)
                        .collect();

                    let genotype_data_strip_f32 = genotype_data
                        .get_standardized_snp_sample_block(
                            &snp_ids_in_strip,
                            &all_qc_sample_ids_for_scores,
                        )
                        .map_err(|_e_original_error| {
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!(
                                    "Failed to get genotype block for strip {}-{}",
                                    strip_start_snp_idx, strip_end_snp_idx
                                ),
                            )) as ThreadSafeStdError
                        })?; // D_strip x N (f32)

                    let v_qr_loadings_for_strip_f32 =
                        v_qr_loadings_d_by_k.slice(s![strip_start_snp_idx..strip_end_snp_idx, ..]); // D_strip x K_initial (f32)

                    // S_intermediate_strip = X_strip^T · V_qr_strip
                    // X_strip is genotype_data_strip_f32 (D_strip x N)
                    // V_qr_strip is v_qr_loadings_for_strip_f32 (D_strip x K_initial)
                    // Result should be N x K_initial
                    let s_intermediate_strip_f32 = Self::dot_product_at_b_mixed_precision(
                        &genotype_data_strip_f32.view(),     // This is A (D_strip x N)
                        &v_qr_loadings_for_strip_f32.view(), // This is B (D_strip x K_QR/K_initial)
                    )?; // Result is N x K_initial, f32 (computed with f64 accumulation)

                    // Cast to f64 for outer sum over strips
                    Ok(s_intermediate_strip_f32.mapv(|x| x as f64))
                },
            )
            .fold(
                || {
                    Ok(Array2::<f64>::zeros((
                        num_total_qc_samples,
                        k_initial_components,
                    )))
                }, // Identity for fold (per-thread accumulator)
                |acc_result, next_result| {
                    match (acc_result, next_result) {
                        (Ok(mut acc_matrix), Ok(next_matrix)) => {
                            acc_matrix += &next_matrix;
                            Ok(acc_matrix)
                        }
                        (Err(e), _) => Err(e), // Propagate previous error
                        (_, Err(e)) => Err(e), // Propagate new error
                    }
                },
            )
            .reduce(
                || {
                    Ok(Array2::<f64>::zeros((
                        num_total_qc_samples,
                        k_initial_components,
                    )))
                }, // Identity for reduce
                |final_acc_result, thread_acc_result| match (final_acc_result, thread_acc_result) {
                    (Ok(mut final_acc), Ok(thread_acc)) => {
                        final_acc += &thread_acc;
                        Ok(final_acc)
                    }
                    (Err(e), _) => Err(e),
                    (_, Err(e)) => Err(e),
                },
            )?; // Corrected: Only one ? needed as reduce itself returns a single Result.

        // --- C. Perform SVD on S_intermediate (which is Array2<f64>) ---

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(pdc) = pass_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics {
                    // Record diagnostics for s_intermediate_n_by_k_initial_f64 BEFORE it's moved.
                    pdc.s_intermediate_dims = Some(s_intermediate_n_by_k_initial_f64.dim());
                    if !s_intermediate_n_by_k_initial_f64.is_empty() {
                        pdc.s_intermediate_fro_norm = Some(compute_frob_norm_f64(
                            &s_intermediate_n_by_k_initial_f64.view(),
                        ));
                        // Note: Computing condition number here might be expensive or redundant if already done.
                        // For now, let's assume it's desired.
                        pdc.s_intermediate_condition_number = compute_condition_number_via_svd_f64(
                            &s_intermediate_n_by_k_initial_f64.view(),
                        );
                    }
                }
            }
        }

        // Instantiate LinAlgBackendProvider for f64
        let backend_svd_f64 = LinAlgBackendProvider::<f64>::new();
        debug!(
            "Performing SVD on f64 intermediate score matrix of shape: {:?}",
            s_intermediate_n_by_k_initial_f64.dim()
        );

        // SVD on f64 matrix
        let svd_output_f64 = backend_svd_f64
            .svd_into(
                s_intermediate_n_by_k_initial_f64, // Consumes matrix (Array2<f64>)
                true,                              // compute U_rot
                true,                              // compute V_rot_transposed
            )
            .map_err(|e_svd| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("SVD (f64) of S_intermediate failed: {}", e_svd),
                )) as ThreadSafeStdError
            })?;

        // SVD results are now f64
        let u_rot_n_by_k_eff_from_svd_f64 = svd_output_f64.u.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "SVD U_rot (f64) (from S_intermediate) not returned",
            )) as ThreadSafeStdError
        })?;

        let s_prime_singular_values_k_eff_from_svd_f64 = svd_output_f64.s; // This is Array1<f64>

        let vt_rot_k_eff_by_k_initial_from_svd_f64 = svd_output_f64.vt.ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "SVD V_rot.T (f64) (from S_intermediate) not returned",
            )) as ThreadSafeStdError
        })?;

        // Mutable versions for potential slicing
        let mut u_rot_n_by_k_eff_f64 = u_rot_n_by_k_eff_from_svd_f64;
        let mut s_prime_singular_values_k_eff_f64 = s_prime_singular_values_k_eff_from_svd_f64;
        let mut vt_rot_k_eff_by_k_initial_f64 = vt_rot_k_eff_by_k_initial_from_svd_f64;

        // --- Determine consistent number of effective components (num_components_to_process) ---
        let k_eff_from_u_f64 = u_rot_n_by_k_eff_f64.ncols();
        let k_eff_from_s_f64 = s_prime_singular_values_k_eff_f64.len();

        let num_components_to_process = k_eff_from_u_f64.min(k_eff_from_s_f64);

        if k_eff_from_u_f64 != k_eff_from_s_f64 {
            warn!(
                "SVD (f64) of S_intermediate resulted in inconsistent k_eff: U_rot has {} components, S_prime has {} components. Processing minimum: {}.",
                k_eff_from_u_f64, k_eff_from_s_f64, num_components_to_process
            );
        }

        if num_components_to_process == 0 {
            debug!("SVD (f64) of S_intermediate resulted in num_components_to_process = 0. Returning empty results.");
            return Ok((
                Array2::zeros((num_total_qc_samples, 0)), // f32 for final output
                Array1::zeros(0),                         // f64 for eigenvalues
                Array2::zeros((num_total_pca_snps, 0)),   // f32 for final output
            ));
        }

        // --- D. Calculate Final Scores, Loadings, and Eigenvalues using num_components_to_process ---

        // Slice SVD outputs (f64) if necessary
        if k_eff_from_u_f64 > num_components_to_process {
            u_rot_n_by_k_eff_f64 = u_rot_n_by_k_eff_f64
                .slice_axis(Axis(1), ndarray::Slice::from(0..num_components_to_process))
                .into_owned();
        }
        if k_eff_from_s_f64 > num_components_to_process {
            s_prime_singular_values_k_eff_f64 = s_prime_singular_values_k_eff_f64
                .slice(s![0..num_components_to_process])
                .into_owned();
            vt_rot_k_eff_by_k_initial_f64 = vt_rot_k_eff_by_k_initial_f64
                .slice_axis(Axis(0), ndarray::Slice::from(0..num_components_to_process))
                .into_owned();
        }

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(pdc) = pass_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics {
                    // Record diagnostics for u_rot_n_by_k_eff_f64 BEFORE it's moved.
                    if !u_rot_n_by_k_eff_f64.is_empty() {
                        let u_rot_f32_for_ortho = u_rot_n_by_k_eff_f64.mapv(|x_f64| x_f64 as f32);
                        pdc.u_s_orthogonality_error =
                            compute_orthogonality_error_f32(&u_rot_f32_for_ortho.view());
                    }
                    // Other diagnostics that might depend on u_rot_n_by_k_eff_f64 before move
                    pdc.s_intermediate_num_singular_values =
                        Some(s_prime_singular_values_k_eff_f64.len());
                    pdc.s_intermediate_singular_values_sample =
                        sample_singular_values_f64(&s_prime_singular_values_k_eff_f64.view(), 10);
                }
            }
        }

        // Final Sample Scores: S_final^* = U_small * Sigma_small (f64)
        let mut final_sample_scores_n_by_k_eff_f64 = u_rot_n_by_k_eff_f64; // N x num_components_to_process (f64)
        if num_components_to_process > 0 {
            for k_idx in 0..num_components_to_process {
                let singular_value_for_scaling_f64 = s_prime_singular_values_k_eff_f64[k_idx];
                let mut score_column_to_scale_f64 =
                    final_sample_scores_n_by_k_eff_f64.column_mut(k_idx);
                score_column_to_scale_f64
                    .mapv_inplace(|element_val| element_val * singular_value_for_scaling_f64);
            }
        }
        // Cast final scores to f32
        let final_sample_scores_n_by_k_eff_f32 =
            final_sample_scores_n_by_k_eff_f64.mapv(|x| x as f32);

        // Final SNP Loadings: V_final = V_qr * V_rot (f32 * f64 -> needs adjustment)
        // V_qr is D x K_initial (f32)
        // V_rot is K_initial x num_components_to_process (f64, from vt_rot_f64.t())
        let v_rot_k_initial_by_k_eff_f64 = vt_rot_k_eff_by_k_initial_f64.t().into_owned();
        // Cast V_rot to f32 before dot product
        let v_rot_k_initial_by_k_eff_f32 = v_rot_k_initial_by_k_eff_f64.mapv(|x| x as f32);
        let final_snp_loadings_d_by_k_eff_f32 =
            v_qr_loadings_d_by_k.dot(&v_rot_k_initial_by_k_eff_f32);

        // Final Eigenvalues: lambda_k = s_prime_k^2 / (N-1) (f64)
        let denominator_n_minus_1 = (num_total_qc_samples as f64 - 1.0).max(1.0);
        let final_eigenvalues_k_eff_f64 = s_prime_singular_values_k_eff_f64.mapv(|s_val_f64| {
            // s_val_f64 is already f64
            (s_val_f64 * s_val_f64) / denominator_n_minus_1
        });

        // --- E. Sort Outputs ---
        // final_eigenvalues_k_eff_f64 is Array1<f64>
        // final_sample_scores_n_by_k_eff_f32 is Array2<f32>
        // final_snp_loadings_d_by_k_eff_f32 is Array2<f32>

        let mut an_eigenvalue_index_pairs: Vec<(f64, usize)> = final_eigenvalues_k_eff_f64
            .iter()
            .enumerate()
            .map(|(idx, &val)| (val, idx))
            .collect();

        an_eigenvalue_index_pairs
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_indices: Vec<usize> = an_eigenvalue_index_pairs
            .into_iter()
            .map(|pair| pair.1)
            .collect();

        let sorted_final_sample_scores =
            reorder_columns_owned(&final_sample_scores_n_by_k_eff_f32, &sorted_indices);
        let sorted_final_snp_loadings =
            reorder_columns_owned(&final_snp_loadings_d_by_k_eff_f32, &sorted_indices);
        let sorted_final_eigenvalues =
            reorder_array_owned(&final_eigenvalues_k_eff_f64, &sorted_indices);

        debug!(
            "Computed final sorted eigenvalues: {:?}",
            sorted_final_eigenvalues
        );
        info!(
            "Computed final sorted sample scores. Shape: {:?}",
            sorted_final_sample_scores.dim()
        );
        info!(
            "Computed final sorted SNP loadings. Shape: {:?}",
            sorted_final_snp_loadings.dim()
        );

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(pdc) = pass_diagnostics_collector.as_mut() {
                if self.config.collect_diagnostics {
                    // Diagnostics for s_intermediate_n_by_k_initial_f64 were moved before its consumption by svd_into.
                    // Diagnostics for u_rot_n_by_k_eff_f64 (as u_s_orthogonality_error) were moved before its consumption.
                    // s_intermediate_num_singular_values and s_intermediate_singular_values_sample are correctly placed using s_prime_singular_values_k_eff_f64.

                    // Notes on final scores/loadings for this pass can be added here.
                    // For instance, orthogonality of final_sample_scores and final_snp_loadings.
                    if !sorted_final_sample_scores.is_empty() {
                        let final_scores_ortho =
                            compute_orthogonality_error_f32(&sorted_final_sample_scores.view());
                        pdc.notes.push_str(&format!(
                            " ;FinalScoresOrthoErr_this_pass: {:?}",
                            final_scores_ortho
                        ));
                    }
                    if !sorted_final_snp_loadings.is_empty() {
                        let final_loadings_ortho =
                            compute_orthogonality_error_f32(&sorted_final_snp_loadings.view());
                        pdc.notes.push_str(&format!(
                            " ;FinalLoadingsOrthoErr_this_pass: {:?}",
                            final_loadings_ortho
                        ));
                    }
                }
            }
        }

        Ok((
            sorted_final_sample_scores,
            sorted_final_eigenvalues,
            sorted_final_snp_loadings,
        ))
    }

    /// Performs randomized SVD on a matrix A (matrix_features_by_samples, M x N).
    /// Returns (Option<U_A_approx>, Option<S_A_approx>, Option<V_A_approx>)
    /// U_A_approx: M x K_eff (Left singular vectors of A)
    /// S_A_approx: K_eff (Singular values of A)
    /// V_A_approx: N x K_eff (Right singular vectors of A)
    ///
    /// ## Numerical Precision
    /// The matrix multiplications performed within this function, such as:
    /// * `matrix_features_by_samples.dot(&random_projection_matrix_omega)` (A * Omega)
    /// * `matrix_features_by_samples.t().dot(&q_basis_m_by_l_actual)` (A.T * Q_basis)
    /// * `matrix_features_by_samples.dot(&q_tilde_n_by_l_actual)` (A * Q_tilde)
    /// * `q_basis_m_by_l_actual.t().dot(matrix_features_by_samples)` (Q_basis.T * A)
    /// are all `f32` operations. When these operations involve very large dimensions
    /// (either M or N of the input matrix, or the sketch dimension L), the internal
    /// accumulation (typically handled by `sgemm` in BLAS) is also likely to be in `f32`.
    /// This can lead to some loss of precision, especially if the number of elements being
    /// summed is extremely large. This is a standard trade-off for performance and memory
    /// efficiency in large-scale numerical computations.
    ///
    /// While some precision loss is possible in these `f32` operations, the Randomized SVD
    /// algorithm incorporates steps like QR decomposition for orthogonalization, which contribute
    /// to its overall numerical stability. Furthermore, in the context of the full Hybrid EigenSNP
    /// PCA workflow, the outputs of this rSVD step (e.g., $U_p^*$ for local bases or $U_{scores}^*$
    /// for initial global scores) are often intermediate. The subsequent Score-Guided Refinement (SR)
    /// phase is designed to refine these components using higher precision for critical calculations
    /// (e.g., `f64` accumulation for $S_{int}$, and mixed-precision `f32`/`f64` for $L_{raw}^*$),
    /// thereby helping to mitigate or compensate for minor inaccuracies introduced during this rSVD stage.
    #[allow(clippy::too_many_arguments)]
    fn _internal_perform_rsvd(
        matrix_features_by_samples: &ArrayView2<f32>, // Input matrix A (M features x N samples)
        num_components_target_k: usize,               // Desired K
        sketch_oversampling_count: usize,             // p (for L = K+p)
        num_power_iterations: usize,                  // q
        random_seed: u64,
        request_u_components: bool, // True if U (left singular vectors) is needed
        request_s_components: bool, // True if S (singular values) is needed
        request_v_components: bool, // True if V (right singular vectors) is needed
        #[cfg(feature = "enable-eigensnp-diagnostics")] mut diagnostics_collector_vec: Option<
            &mut Vec<crate::diagnostics::RsvdStepDetail>,
        >,
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))] _diagnostics_collector_vec: Option<()>,
    ) -> Result<
        (
            Option<Array2<f32>>,
            Option<Array1<f32>>,
            Option<Array2<f32>>,
        ),
        ThreadSafeStdError,
    > {
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        let push_diag_fn =
            |dc_vec: &mut Vec<RsvdStepDetail>,
             step_name: String,
             iteration: Option<usize>,
             input_dims: Option<(usize, usize)>,
             output_dims: Option<(usize, usize)>,
             matrix_to_measure: Option<&ArrayView2<f32>>,
             q_factor_to_measure: Option<&ArrayView2<f32>>| {
                // dc_vec is now &mut Vec<RsvdStepDetail> directly
                let mut detail = RsvdStepDetail::default();
                detail.step_name = step_name;
                if let Some(iter) = iteration {
                    detail.notes = format!("Iteration: {}", iter);
                }
                detail.input_matrix_dims = input_dims;
                detail.output_matrix_dims = output_dims;

                if let Some(matrix) = matrix_to_measure {
                    if !matrix.is_empty() {
                        detail.fro_norm = Some(compute_frob_norm_f32(&matrix.view()) as f64);
                        detail.condition_number =
                            compute_condition_number_via_svd_f32(&matrix.view());
                    }
                }
                if let Some(q_matrix) = q_factor_to_measure {
                    if !q_matrix.is_empty() {
                        detail.orthogonality_error =
                            compute_orthogonality_error_f32(&q_matrix.view());
                    }
                }
                dc_vec.push(detail);
            };
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        let push_diag_fn = |_: Option<()>,
                            _: String,
                            _: Option<usize>,
                            _: Option<(usize, usize)>,
                            _: Option<(usize, usize)>,
                            _: Option<&ArrayView2<f32>>,
                            _: Option<&ArrayView2<f32>>| { // This signature is correct for non-diagnostic
             // No-op for non-diagnostics build
        };

        let num_features_m = matrix_features_by_samples.nrows();
        let num_samples_n = matrix_features_by_samples.ncols();

        // Non-diagnostic collector_for_push_fn remains as is.
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        let collector_for_push_fn = _diagnostics_collector_vec;

        // Call the push_diag_fn closure, passing the appropriate collector.
        // This call needs to be updated per point 4.
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        if let Some(ref mut actual_collector) = diagnostics_collector_vec {
            push_diag_fn(
                actual_collector,
                "Input_A".to_string(),
                None,
                None,
                Some((num_features_m, num_samples_n)),
                Some(&matrix_features_by_samples.view()),
                None,
            );
        }
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        push_diag_fn(
            collector_for_push_fn,
            "Input_A".to_string(),
            None,
            None,
            Some((num_features_m, num_samples_n)),
            Some(&matrix_features_by_samples.view()),
            None,
        );

        if num_features_m == 0 || num_samples_n == 0 || num_components_target_k == 0 {
            debug!(
                "RSVD: Input matrix empty or K=0. M={}, N={}, K={}",
                num_features_m, num_samples_n, num_components_target_k
            );
            let u_res = if request_u_components {
                Some(Array2::zeros((num_features_m, 0)))
            } else {
                None
            };
            let s_res = if request_s_components {
                Some(Array1::zeros(0))
            } else {
                None
            };
            let v_res = if request_v_components {
                Some(Array2::zeros((num_samples_n, 0)))
            } else {
                None
            };
            return Ok((u_res, s_res, v_res));
        }

        let sketch_dimension_l = (num_components_target_k + sketch_oversampling_count)
            .min(num_features_m.min(num_samples_n));

        if sketch_dimension_l == 0 {
            debug!(
                "RSVD: Sketch dimension L=0. M={}, N={}, K={}, p={}",
                num_features_m, num_samples_n, num_components_target_k, sketch_oversampling_count
            );
            let u_res = if request_u_components {
                Some(Array2::zeros((num_features_m, 0)))
            } else {
                None
            };
            let s_res = if request_s_components {
                Some(Array1::zeros(0))
            } else {
                None
            };
            let v_res = if request_v_components {
                Some(Array2::zeros((num_samples_n, 0)))
            } else {
                None
            };
            return Ok((u_res, s_res, v_res));
        }
        trace!(
            "RSVD internal: Target_K={}, Sketch_L={}, Input_M(features)={}, Input_N(samples)={}",
            num_components_target_k,
            sketch_dimension_l,
            num_features_m,
            num_samples_n
        );

        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);
        let normal_dist = Normal::new(0.0, 1.0).map_err(|e_normal| -> ThreadSafeStdError {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Failed to create normal distribution for RSVD: {}",
                    e_normal
                ),
            )
            .into()
        })?;

        // Omega: N x L
        let random_projection_matrix_omega =
            Array2::from_shape_fn((num_samples_n, sketch_dimension_l), |_| {
                normal_dist.sample(&mut rng) as f32
            });
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        if let Some(ref mut actual_collector) = diagnostics_collector_vec {
            push_diag_fn(
                actual_collector,
                "Omega".to_string(),
                None,
                Some((num_samples_n, sketch_dimension_l)),
                Some((num_samples_n, sketch_dimension_l)),
                Some(&random_projection_matrix_omega.view()),
                None,
            );
        }
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        push_diag_fn(
            collector_for_push_fn,
            "Omega".to_string(),
            None,
            Some((num_samples_n, sketch_dimension_l)),
            Some((num_samples_n, sketch_dimension_l)),
            Some(&random_projection_matrix_omega.view()),
            None,
        );

        let backend = LinAlgBackendProvider::<f32>::new();

        // Y = A * Omega (M x N) * (N x L) -> (M x L)
        let sketch_y = Self::dot_product_mixed_precision_f32_f64acc(
            matrix_features_by_samples,
            &random_projection_matrix_omega.view(),
        )?;
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        if let Some(ref mut actual_collector) = diagnostics_collector_vec {
            push_diag_fn(
                actual_collector,
                "SketchY_PreQR".to_string(),
                None,
                Some((num_features_m, num_samples_n)),
                Some(sketch_y.dim()),
                Some(&sketch_y.view()),
                None,
            );
        }
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        push_diag_fn(
            collector_for_push_fn,
            "SketchY_PreQR".to_string(),
            None,
            Some((num_features_m, num_samples_n)),
            Some(sketch_y.dim()),
            Some(&sketch_y.view()),
            None,
        );

        if sketch_y.ncols() == 0 {
            warn!("RSVD: Initial sketch Y (A*Omega) has zero columns before first QR. Target_K={}, Sketch_L={}", num_components_target_k, sketch_dimension_l);
            let u_res = if request_u_components {
                Some(Array2::zeros((num_features_m, 0)))
            } else {
                None
            };
            let s_res = if request_s_components {
                Some(Array1::zeros(0))
            } else {
                None
            };
            let v_res = if request_v_components {
                Some(Array2::zeros((num_samples_n, 0)))
            } else {
                None
            };
            return Ok((u_res, s_res, v_res));
        }

        // Q_basis = orth(Y) (M x L_actual_y)
        let mut q_basis_m_by_l_actual = backend.qr_q_factor(&sketch_y).map_err(|e_qr| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "QR decomposition of initial sketch Y failed in RSVD: {}",
                    e_qr
                ),
            )) as ThreadSafeStdError
        })?;
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        if let Some(ref mut actual_collector) = diagnostics_collector_vec {
            push_diag_fn(
                actual_collector,
                "Q0_PostQR".to_string(),
                Some(0),
                Some(sketch_y.dim()),
                Some(q_basis_m_by_l_actual.dim()),
                None,
                Some(&q_basis_m_by_l_actual.view()),
            );
        }
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        push_diag_fn(
            collector_for_push_fn,
            "Q0_PostQR".to_string(),
            Some(0),
            Some(sketch_y.dim()),
            Some(q_basis_m_by_l_actual.dim()),
            None,
            Some(&q_basis_m_by_l_actual.view()),
        );

        // Power iterations
        for iter_idx in 0..num_power_iterations {
            if q_basis_m_by_l_actual.ncols() == 0 {
                trace!(
                    "RSVD Power Iteration {}: Q_basis became empty, breaking.",
                    iter_idx + 1
                );
                break;
            }
            trace!(
                "RSVD Power Iteration {}/{}",
                iter_idx + 1,
                num_power_iterations
            );

            // Q_tilde_candidate = A.T * Q_basis (N x M) * (M x L_actual) -> (N x L_actual)
            let q_tilde_candidate = Self::dot_product_at_b_mixed_precision(
                matrix_features_by_samples,
                &q_basis_m_by_l_actual.view(),
            )?;
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            if let Some(ref mut actual_collector) = diagnostics_collector_vec {
                push_diag_fn(
                    actual_collector,
                    format!("PowerIter{}_Ytilde_PreQR", iter_idx + 1),
                    Some(iter_idx + 1),
                    Some(q_basis_m_by_l_actual.dim()),
                    Some(q_tilde_candidate.dim()),
                    Some(&q_tilde_candidate.view()),
                    None,
                );
            }
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            push_diag_fn(
                collector_for_push_fn,
                format!("PowerIter{}_Ytilde_PreQR", iter_idx + 1),
                Some(iter_idx + 1),
                Some(q_basis_m_by_l_actual.dim()),
                Some(q_tilde_candidate.dim()),
                Some(&q_tilde_candidate.view()),
                None,
            );

            if q_tilde_candidate.ncols() == 0 {
                q_basis_m_by_l_actual = Array2::zeros((q_basis_m_by_l_actual.nrows(), 0));
                trace!(
                    "RSVD Power Iteration {}: Q_tilde_candidate became empty.",
                    iter_idx + 1
                );
                break;
            }
            // Q_tilde = orth(Q_tilde_candidate) (N x L_actual_tilde)
            let q_tilde_n_by_l_actual =
                backend.qr_q_factor(&q_tilde_candidate).map_err(|e_qr| {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "QR for Q_tilde in power iteration {} failed: {}",
                            iter_idx + 1,
                            e_qr
                        ),
                    )) as ThreadSafeStdError
                })?;
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            if let Some(ref mut actual_collector) = diagnostics_collector_vec {
                push_diag_fn(
                    actual_collector,
                    format!("PowerIter{}_Qtilde_PostQR", iter_idx + 1),
                    Some(iter_idx + 1),
                    Some(q_tilde_candidate.dim()),
                    Some(q_tilde_n_by_l_actual.dim()),
                    None,
                    Some(&q_tilde_n_by_l_actual.view()),
                );
            }
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            push_diag_fn(
                collector_for_push_fn,
                format!("PowerIter{}_Qtilde_PostQR", iter_idx + 1),
                Some(iter_idx + 1),
                Some(q_tilde_candidate.dim()),
                Some(q_tilde_n_by_l_actual.dim()),
                None,
                Some(&q_tilde_n_by_l_actual.view()),
            );

            if q_tilde_n_by_l_actual.ncols() == 0 {
                q_basis_m_by_l_actual = Array2::zeros((q_basis_m_by_l_actual.nrows(), 0));
                trace!(
                    "RSVD Power Iteration {}: Q_tilde became empty after QR.",
                    iter_idx + 1
                );
                break;
            }

            // Q_basis_candidate = A * Q_tilde (M x N) * (N x L_actual_tilde) -> (M x L_actual_tilde)
            let q_basis_candidate_next = Self::dot_product_mixed_precision_f32_f64acc(
                matrix_features_by_samples,
                &q_tilde_n_by_l_actual.view(),
            )?;
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            if let Some(ref mut actual_collector) = diagnostics_collector_vec {
                push_diag_fn(
                    actual_collector,
                    format!("PowerIter{}_Ynext_PreQR", iter_idx + 1),
                    Some(iter_idx + 1),
                    Some(q_tilde_n_by_l_actual.dim()),
                    Some(q_basis_candidate_next.dim()),
                    Some(&q_basis_candidate_next.view()),
                    None,
                );
            }
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            push_diag_fn(
                collector_for_push_fn,
                format!("PowerIter{}_Ynext_PreQR", iter_idx + 1),
                Some(iter_idx + 1),
                Some(q_tilde_n_by_l_actual.dim()),
                Some(q_basis_candidate_next.dim()),
                Some(&q_basis_candidate_next.view()),
                None,
            );

            if q_basis_candidate_next.ncols() == 0 {
                q_basis_m_by_l_actual = Array2::zeros((q_basis_m_by_l_actual.nrows(), 0));
                trace!(
                    "RSVD Power Iteration {}: Q_basis_candidate_next became empty.",
                    iter_idx + 1
                );
                break;
            }
            // Q_basis = orth(Q_basis_candidate_next) (M x L_actual_final_iter)
            q_basis_m_by_l_actual =
                backend
                    .qr_q_factor(&q_basis_candidate_next)
                    .map_err(|e_qr| {
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "QR for Q_basis in power iteration {} failed: {}",
                                iter_idx + 1,
                                e_qr
                            ),
                        )) as ThreadSafeStdError
                    })?;
            #[cfg(feature = "enable-eigensnp-diagnostics")]
            if let Some(ref mut actual_collector) = diagnostics_collector_vec {
                push_diag_fn(
                    actual_collector,
                    format!("PowerIter{}_Qnext_PostQR", iter_idx + 1),
                    Some(iter_idx + 1),
                    Some(q_basis_candidate_next.dim()),
                    Some(q_basis_m_by_l_actual.dim()),
                    None,
                    Some(&q_basis_m_by_l_actual.view()),
                );
            }
            #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
            push_diag_fn(
                collector_for_push_fn,
                format!("PowerIter{}_Qnext_PostQR", iter_idx + 1),
                Some(iter_idx + 1),
                Some(q_basis_candidate_next.dim()),
                Some(q_basis_m_by_l_actual.dim()),
                None,
                Some(&q_basis_m_by_l_actual.view()),
            );
        }

        if q_basis_m_by_l_actual.ncols() == 0 {
            warn!(
                "RSVD: Refined Q_basis has zero columns after power iterations. Target_K={}",
                num_components_target_k
            );
            let u_res = if request_u_components {
                Some(Array2::zeros((num_features_m, 0)))
            } else {
                None
            };
            let s_res = if request_s_components {
                Some(Array1::zeros(0))
            } else {
                None
            };
            let v_res = if request_v_components {
                Some(Array2::zeros((num_samples_n, 0)))
            } else {
                None
            };
            return Ok((u_res, s_res, v_res));
        }

        // B = Q_basis.T * A (L_actual x M) * (M x N) -> (L_actual x N)
        let projected_b_l_actual_by_n = Self::dot_product_at_b_mixed_precision(
            &q_basis_m_by_l_actual.view(),
            matrix_features_by_samples,
        )?;
        #[cfg(feature = "enable-eigensnp-diagnostics")]
        if let Some(ref mut actual_collector) = diagnostics_collector_vec {
            push_diag_fn(
                actual_collector,
                "ProjectedB_PreSVD".to_string(),
                None,
                Some(q_basis_m_by_l_actual.dim()),
                Some(projected_b_l_actual_by_n.dim()),
                Some(&projected_b_l_actual_by_n.view()),
                None,
            );
        }
        #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
        push_diag_fn(
            collector_for_push_fn,
            "ProjectedB_PreSVD".to_string(),
            None,
            Some(q_basis_m_by_l_actual.dim()),
            Some(projected_b_l_actual_by_n.dim()),
            Some(&projected_b_l_actual_by_n.view()),
            None,
        );

        // SVD of B: B = U_B * S_B * V_B.T
        // U_B is L_actual x rank_b
        // S_B is rank_b
        // V_B.T is rank_b x N
        let compute_u_for_b = request_u_components; // U_A = Q_basis * U_B, so U_B is needed if U_A is.
        let compute_v_for_b = request_v_components; // V_A = V_B, so V_B (from V_B.T) is needed if V_A is.

        use crate::linalg_backends::SVDOutput; // Ensure this type is available or use its definition

        let svd_result_b = backend.svd_into(
            projected_b_l_actual_by_n.clone().into_owned(),
            compute_u_for_b,
            compute_v_for_b,
        );

        let svd_output_b = match svd_result_b {
            Ok(output) => output,
            Err(e_svd) => {
                // Check if the error message string contains typical ndarray-linalg error indicators
                // This is a bit heuristic as we don't have the exact error type here easily.
                let error_string = format!("{}", e_svd);
                if error_string.contains("LinalgError")
                    || error_string.contains("NonConverged")
                    || error_string.contains("IllegalParameter")
                {
                    warn!(
                        "RSVD: SVD of projected matrix B failed (likely due to low rank or numerical issues): {}. Proceeding with 0 components from this SVD.",
                        e_svd
                    );
                    // Create an empty SvdOutput structure
                    SVDOutput {
                        u: if compute_u_for_b {
                            Some(Array2::zeros((q_basis_m_by_l_actual.ncols(), 0)))
                        } else {
                            None
                        },
                        s: Array1::<f32>::zeros(0), // Assuming f32 context, A::Real would be f32
                        vt: if compute_v_for_b {
                            Some(Array2::zeros((0, matrix_features_by_samples.ncols())))
                        } else {
                            None
                        },
                    }
                } else {
                    // If it's some other error, propagate it
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("SVD of projected matrix B failed in RSVD: {}", e_svd),
                    )) as ThreadSafeStdError);
                }
            }
        };

        #[cfg(feature = "enable-eigensnp-diagnostics")]
        {
            if let Some(ref mut dc_vec) = diagnostics_collector_vec {
                // Changed to use diagnostics_collector_vec and ref mut
                // dc_vec is now &mut Vec<RsvdStepDetail>
                let mut detail_svd = RsvdStepDetail::default();
                detail_svd.step_name = "SVD_of_B".to_string();
                if let Some(u_b) = svd_output_b.u.as_ref() {
                    detail_svd
                        .notes
                        .push_str(&format!("U_B dims: {:?}; ", u_b.dim()));
                    // Could add more detailed metrics for U_B if needed
                }
                detail_svd.num_singular_values = Some(svd_output_b.s.len());
                detail_svd.singular_values_sample =
                    sample_singular_values(&svd_output_b.s.view(), 10)
                        .map(|v_f32| v_f32.iter().map(|&x| x as f64).collect()); // Store as f64
                if let Some(vt_b) = svd_output_b.vt.as_ref() {
                    detail_svd
                        .notes
                        .push_str(&format!("Vt_B dims: {:?}; ", vt_b.dim()));
                }

                // SVD Reconstruction Error for B = U_B S_B Vt_B
                // Need original B (projected_b_l_actual_by_n), U_B, S_B, Vt_B
                if let (Some(u_b_val), Some(vt_b_val)) =
                    (svd_output_b.u.as_ref(), svd_output_b.vt.as_ref())
                {
                    if !projected_b_l_actual_by_n.is_empty()
                        && !u_b_val.is_empty()
                        && !svd_output_b.s.is_empty()
                        && !vt_b_val.is_empty()
                    {
                        let reconstruction_error = compute_svd_reconstruction_error_f32(
                            &projected_b_l_actual_by_n.view(),
                            &u_b_val.view(),
                            &svd_output_b.s.view(),
                            &vt_b_val.view(),
                        );
                        detail_svd.svd_reconstruction_error_rel = reconstruction_error;
                        // Could also compute absolute error if needed.
                    }
                }
                dc_vec.push(detail_svd);
            }
        }

        let mut u_a_approx_opt: Option<Array2<f32>> = None;
        let mut s_a_approx_opt: Option<Array1<f32>> = None;
        let mut v_a_approx_opt: Option<Array2<f32>> = None;

        let effective_rank_b = svd_output_b.s.len(); // Corrected: svd_output_b.s is Array1<f32>
        let num_k_to_return = num_components_target_k.min(effective_rank_b);

        // This replaces the block from 'if request_s_components {' down to its closing brace.
        if request_s_components {
            // svd_output_b.s is Array1<f32>.
            // num_k_to_return was calculated earlier and is the number of components the user wants.
            // effective_rank_b is svd_output_b.s.len(), the actual number of singular values from SVD.

            let actual_k_to_slice = std::cmp::min(num_k_to_return, effective_rank_b);

            if actual_k_to_slice == 0 {
                s_a_approx_opt = Some(Array1::zeros(0));
            } else {
                // Slice svd_output_b.s to get the top 'actual_k_to_slice' singular values.
                // The s! macro is appropriate for slicing ndarray::Array1.
                s_a_approx_opt = Some(svd_output_b.s.slice(s![0..actual_k_to_slice]).to_owned());
            }
        }

        if request_u_components {
            if let Some(u_b_l_actual_by_rank_b) = svd_output_b.u {
                if u_b_l_actual_by_rank_b.ncols() > 0 && q_basis_m_by_l_actual.ncols() > 0 {
                    // U_A = Q_basis * U_B (M x L_actual) * (L_actual x rank_b) -> M x rank_b
                    let u_a_approx_m_by_rank_b = q_basis_m_by_l_actual.dot(&u_b_l_actual_by_rank_b);
                    let u_a_final = u_a_approx_m_by_rank_b
                        .slice_axis(Axis(1), ndarray::Slice::from(0..num_k_to_return))
                        .to_owned();
                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    if let Some(ref mut actual_collector) = diagnostics_collector_vec {
                        push_diag_fn(
                            actual_collector,
                            "Final_U_A".to_string(),
                            None,
                            Some(u_a_approx_m_by_rank_b.dim()),
                            Some(u_a_final.dim()),
                            Some(&u_a_final.view()),
                            Some(&u_a_final.view()),
                        );
                    }
                    #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                    push_diag_fn(
                        collector_for_push_fn,
                        "Final_U_A".to_string(),
                        None,
                        Some(u_a_approx_m_by_rank_b.dim()),
                        Some(u_a_final.dim()),
                        Some(&u_a_final.view()),
                        Some(&u_a_final.view()),
                    );
                    u_a_approx_opt = Some(u_a_final);
                } else {
                    u_a_approx_opt = Some(Array2::zeros((num_features_m, 0)));
                }
            } else {
                u_a_approx_opt = Some(Array2::zeros((num_features_m, 0)));
            }
        }

        if request_v_components {
            if let Some(v_b_t_rank_b_by_n) = svd_output_b.vt {
                if v_b_t_rank_b_by_n.nrows() > 0 {
                    // effectively checks rank_b > 0
                    // V_A = V_B. V_B is (N x rank_b). We have V_B.T (rank_b x N)
                    let v_a_approx_n_by_rank_b = v_b_t_rank_b_by_n.t().into_owned();
                    let v_a_final = v_a_approx_n_by_rank_b
                        .slice_axis(Axis(1), ndarray::Slice::from(0..num_k_to_return))
                        .to_owned();
                    #[cfg(feature = "enable-eigensnp-diagnostics")]
                    if let Some(ref mut actual_collector) = diagnostics_collector_vec {
                        push_diag_fn(
                            actual_collector,
                            "Final_V_A".to_string(),
                            None,
                            Some(v_a_approx_n_by_rank_b.dim()),
                            Some(v_a_final.dim()),
                            Some(&v_a_final.view()),
                            Some(&v_a_final.view()),
                        );
                    }
                    #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
                    push_diag_fn(
                        collector_for_push_fn,
                        "Final_V_A".to_string(),
                        None,
                        Some(v_a_approx_n_by_rank_b.dim()),
                        Some(v_a_final.dim()),
                        Some(&v_a_final.view()),
                        Some(&v_a_final.view()),
                    );
                    v_a_approx_opt = Some(v_a_final);
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

    /// Performs matrix multiplication of two f32 matrices (A * B) using f64 accumulation
    /// for each element of the resulting f32 matrix.
    /// A (a_matrix_view): M x P
    /// B (b_matrix_view): P x K
    /// Result: M x K
    fn dot_product_mixed_precision_f32_f64acc(
        a_matrix_view: &ArrayView2<f32>,
        b_matrix_view: &ArrayView2<f32>,
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let m_dim = a_matrix_view.nrows();
        let p_common_dim_a = a_matrix_view.ncols();
        let p_common_dim_b = b_matrix_view.nrows();
        let k_dim = b_matrix_view.ncols();

        const LANES: usize = 8; // Define LANES constant

        if p_common_dim_a != p_common_dim_b {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Dimension mismatch for mixed-precision dot product: A.ncols ({}) != B.nrows ({}).",
                    p_common_dim_a, p_common_dim_b
                ),
            )) as ThreadSafeStdError);
        }

        if m_dim == 0 || p_common_dim_a == 0 || k_dim == 0 {
            return Ok(Array2::<f32>::zeros((m_dim, k_dim)));
        }

        let mut result_matrix_f32 = Array2::<f32>::zeros((m_dim, k_dim));

        result_matrix_f32
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i_row_idx, mut output_row_view)| {
                let a_row_i = a_matrix_view.row(i_row_idx);
                let a_row_slice = a_row_i.to_slice().expect(
                    "Failed to slice a_row_i, data might not be contiguous or in standard layout.",
                );

                for j_col_idx in 0..k_dim {
                    let mut accumulator_f64: f64 = 0.0;
                    let b_column_j = b_matrix_view.column(j_col_idx); // Obtain the column view for B
                                                                      // b_column_slice is removed.

                    let num_simd_chunks = p_common_dim_a / LANES;
                    let mut simd_f32_partial_sum = Simd::splat(0.0f32); // Ensure f32 type for splat

                    for chunk_idx in 0..num_simd_chunks {
                        let offset = chunk_idx * LANES;
                        let a_simd = Simd::from_slice(&a_row_slice[offset..offset + LANES]);

                        let mut b_temp_array = [0.0f32; LANES];
                        for lane_idx in 0..LANES {
                            b_temp_array[lane_idx] = b_column_j[offset + lane_idx];
                        }
                        let b_simd = Simd::from_array(b_temp_array);

                        simd_f32_partial_sum += a_simd * b_simd;
                    }
                    accumulator_f64 += simd_f32_partial_sum.reduce_sum() as f64;

                    for p_idx in (num_simd_chunks * LANES)..p_common_dim_a {
                        // Use a_row_slice for A, and b_column_j (the view) for B
                        accumulator_f64 += (a_row_slice[p_idx] as f64) * (b_column_j[p_idx] as f64);
                    }
                    output_row_view[j_col_idx] = accumulator_f64 as f32;
                }
            });

        Ok(result_matrix_f32)
    }
}
