// src/diagnostics.rs
#![cfg(feature = "enable-eigensnp-diagnostics")]

use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::linalg_backends::LinAlgBackendProvider; // SVDOutput might not be needed directly
use serde::{Serialize, Deserialize};
use std::f64::INFINITY;
// use std::fmt; // Not used currently

// --- New Struct Definitions ---

/// Detailed diagnostics for a single step within an rSVD computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RsvdStepDetail {
    pub step_name: String,                      // e.g., "Y_normalization", "Q_computation", "S_svd"
    pub input_matrix_dims: Option<(usize, usize)>, // (rows, cols)
    pub output_matrix_dims: Option<(usize, usize)>, // (rows, cols)
    
    // --- Generic Matrix Metrics (apply to input or output depending on step context) ---
    pub fro_norm: Option<f64>,                  // Frobenius norm
    pub condition_number: Option<f64>,          // Condition number (via SVD, f64 backend)
    
    // --- Orthogonality Metrics (primarily for Q factors) ---
    pub orthogonality_error: Option<f64>,       // ||I - Q^T Q||_F
    
    // --- SVD-Specific Metrics (for steps involving SVD) ---
    pub svd_reconstruction_error_abs: Option<f64>, // ||A - USV^T||_F (absolute error)
    pub svd_reconstruction_error_rel: Option<f64>, // ||A - USV^T||_F / ||A||_F (relative error)
    pub num_singular_values: Option<usize>,     // Total singular values computed
    pub singular_values_sample: Option<Vec<f64>>, // Sample of singular values
    
    pub notes: String,                          // Any additional context or free-form notes
}

/// Diagnostics for the local PCA basis computation within a single block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerBlockLocalBasisDiagnostics {
    pub block_id: String,                       // Identifier for the LD block or segment
    pub rsvd_stages: Vec<RsvdStepDetail>,       // Diagnostics for each rSVD stage applied
    
    // --- Correlation with f64 Ground Truth (if available) ---
    // Absolute Pearson correlation coefficients for each column vector
    pub u_correlation_vs_f64_truth: Option<Vec<f64>>, 
    
    // --- Final Output Metrics for this Block's U_p (local basis) ---
    pub u_p_dims: Option<(usize, usize)>,
    pub u_p_fro_norm: Option<f64>,
    pub u_p_condition_number: Option<f64>,
    pub u_p_orthogonality_error: Option<f64>,
    
    pub notes: String,
}

/// Diagnostics for the global PCA step (on the condensed, standardized matrix).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalPcaDiagnostics {
    pub stage_name: String,                     // e.g., "GlobalPCA"
    pub rsvd_stages: Vec<RsvdStepDetail>,       // Diagnostics for rSVD stages
    
    // --- Correlation with Python's PCA output on initial scores (U_scores_star) ---
    // Used for validating the initial projection against a known Python implementation
    pub initial_scores_correlation_vs_py_truth: Option<Vec<f64>>,
    
    pub notes: String,
}

/// Detailed diagnostics for a single pass of the Subspace Refinement (SR) algorithm.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SrPassDetail {
    pub pass_num: usize,
    
    // Metrics for V_hat (eigenvectors of S^T S) from the previous pass or initial global PCA
    pub v_hat_dims: Option<(usize, usize)>,
    pub v_hat_orthogonality_error: Option<f64>, // Error for V_hat from (S^T S) V_hat = V_hat Lambda
                                                // or Q in V_hat = QR decomposition if applicable
                                                
    // Metrics for the intermediate matrix S_intermediate = C_std @ V_hat
    pub s_intermediate_dims: Option<(usize, usize)>,
    pub s_intermediate_fro_norm: Option<f64>,
    pub s_intermediate_condition_number: Option<f64>,
    
    // SVD of S_intermediate: S_intermediate = U_s S_s V_s^T
    pub s_intermediate_svd_reconstruction_error_abs: Option<f64>,
    pub s_intermediate_svd_reconstruction_error_rel: Option<f64>,
    pub s_intermediate_num_singular_values: Option<usize>,
    pub s_intermediate_singular_values_sample: Option<Vec<f64>>,
    
    // Orthogonality of U_s from SVD of S_intermediate
    pub u_s_orthogonality_error: Option<f64>,
    
    pub notes: String,
}

/// Comprehensive diagnostics for a full PCA run, encompassing all major stages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FullPcaRunDetailedDiagnostics {
    // --- Local Basis Computation ---
    pub per_block_diagnostics: Vec<PerBlockLocalBasisDiagnostics>, // One for each LD block
    
    // --- Condensed Matrix C (after concatenating U_p^T X_p) ---
    pub c_matrix_dims: Option<(usize, usize)>,
    pub c_matrix_fro_norm: Option<f64>,         // Norm of C
    
    // --- Standardized Condensed Matrix C_std ---
    pub c_std_matrix_dims: Option<(usize, usize)>,
    pub c_std_matrix_fro_norm: Option<f64>,     // Norm of C_std
    pub c_std_col_means_sample: Option<Vec<f64>>, // Sample of column means
    pub c_std_col_std_devs_sample: Option<Vec<f64>>, // Sample of column standard deviations
    
    // --- Global PCA Diagnostics (on C_std) ---
    pub global_pca_diag: Option<Box<GlobalPcaDiagnostics>>, // Boxed to manage size
    
    // --- Subspace Refinement (SR) Passes ---
    pub sr_pass_details: Vec<SrPassDetail>,     // One for each SR pass
    
    // --- Final Output Metrics (e.g., final PCs, final Loadings if computed) ---
    // These might be added later or be part of a separate struct if too detailed
    
    pub total_runtime_seconds: Option<f64>,     // Overall runtime for the PCA process
    pub notes: String,                          // High-level notes for the entire run
}


// --- Utility Functions for Metrics ---

/// Computes Frobenius norm for an f32 matrix.
pub fn compute_frob_norm_f32(matrix: &ArrayView2<f32>) -> f32 {
    if matrix.is_empty() {
        return 0.0;
    }
    matrix.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Computes Frobenius norm for an f64 matrix.
pub fn compute_frob_norm_f64(matrix: &ArrayView2<f64>) -> f64 {
    if matrix.is_empty() {
        return 0.0;
    }
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Computes condition number via SVD for an f32 matrix, using f64 backend for SVD.
pub fn compute_condition_number_via_svd_f32(matrix: &ArrayView2<f32>) -> Option<f64> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return None;
    }
    let matrix_f64 = matrix.mapv(|x| x as f64);
    compute_condition_number_via_svd_f64(&matrix_f64.view())
}

/// Computes condition number via SVD for an f64 matrix.
/// Condition number = sigma_max / sigma_min.
pub fn compute_condition_number_via_svd_f64(matrix: &ArrayView2<f64>) -> Option<f64> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return None;
    }

    let backend_f64 = LinAlgBackendProvider::<f64>::new();
    let svd_result = backend_f64.svd_into(matrix.to_owned(), false, false);
    
    let singular_values = match svd_result {
        Ok(output) => output.s,
        Err(_) => return None, // SVD failed
    };

    if singular_values.is_empty() {
        return Some(0.0); // Or None if preferred for no singular values
    }

    let sigma_max = singular_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sigma_min_non_zero = singular_values.iter().cloned().filter(|&s| s > 1e-12).fold(f64::INFINITY, f64::min);

    if sigma_min_non_zero == f64::INFINITY || sigma_min_non_zero <= 1e-12 {
        return Some(INFINITY); // Effectively zero or no non-zero sigma_min
    }
    if sigma_max == f64::NEG_INFINITY { // Should not happen if singular_values not empty
        return Some(0.0); 
    }
    
    Some(sigma_max / sigma_min_non_zero)
}

/// Computes orthogonality error ||I - Q^T Q||_F for an f32 matrix.
pub fn compute_orthogonality_error_f32(q_matrix: &ArrayView2<f32>) -> Option<f64> {
    if q_matrix.nrows() == 0 || q_matrix.ncols() == 0 {
        return None;
    }
    let q_f64 = q_matrix.mapv(|x| x as f64);
    compute_orthogonality_error_f64(&q_f64.view())
}

/// Computes orthogonality error ||I - Q^T Q||_F for an f64 matrix.
pub fn compute_orthogonality_error_f64(q_matrix: &ArrayView2<f64>) -> Option<f64> {
    if q_matrix.nrows() == 0 || q_matrix.ncols() == 0 {
        return None;
    }
    if q_matrix.nrows() < q_matrix.ncols() {
        // This typically means Q is "wide", so Q^T Q would be k x k where k = ncols
        // If Q is orthogonal, Q^T Q = I_k.
        // If Q is "tall" (more common for basis matrices), Q^T Q = I_k still.
        // This check is more of a sanity check; the math should hold.
        // log::warn!("Orthogonality check for matrix with nrows < ncols ({} < {})", q_matrix.nrows(), q_matrix.ncols());
    }

    let qtq = q_matrix.t().dot(q_matrix);
    let identity = Array2::<f64>::eye(qtq.nrows());
    let diff = identity - qtq;
    
    Some(compute_frob_norm_f64(&diff.view()))
}

/// Computes SVD reconstruction error ||A - USV^T||_F / ||A||_F for f32 inputs.
/// Uses f64 for intermediate calculations.
pub fn compute_svd_reconstruction_error_f32(
    original_matrix: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    s_vec: &ArrayView1<f32>,
    vt: &ArrayView2<f32>,
) -> Option<f64> {
    if original_matrix.is_empty() { return None; }

    let original_f64 = original_matrix.mapv(|x| x as f64);
    let u_f64 = u.mapv(|x| x as f64);
    let s_vec_f64 = s_vec.mapv(|x| x as f64);
    let vt_f64 = vt.mapv(|x| x as f64);

    compute_svd_reconstruction_error_f64(
        &original_f64.view(),
        &u_f64.view(),
        &s_vec_f64.view(),
        &vt_f64.view(),
    )
}

/// Computes SVD reconstruction error ||A - USV^T||_F / ||A||_F for f64 inputs.
pub fn compute_svd_reconstruction_error_f64(
    original_matrix: &ArrayView2<f64>,
    u: &ArrayView2<f64>,
    s_vec: &ArrayView1<f64>,
    vt: &ArrayView2<f64>,
) -> Option<f64> {
    if original_matrix.is_empty() { return None; }
    if u.ncols() != s_vec.len() || s_vec.len() != vt.nrows() { return None; } // Dim mismatch
    if u.nrows() != original_matrix.nrows() || vt.ncols() != original_matrix.ncols() { return None; } // Dim mismatch

    let s_diag = Array2::from_diag(s_vec);
    let reconstructed_matrix = u.dot(&s_diag).dot(vt);
    let diff = original_matrix - &reconstructed_matrix; // Use borrow here

    let norm_diff = compute_frob_norm_f64(&diff.view());
    let norm_original = compute_frob_norm_f64(&original_matrix.view());

    if norm_original < 1e-12 { // Original matrix is close to zero
        if norm_diff < 1e-12 { // Diff is also close to zero
            Some(0.0) // Perfect reconstruction of a zero matrix
        } else {
            Some(INFINITY) // Non-zero difference from a zero matrix
        }
    } else {
        Some(norm_diff / norm_original)
    }
}

/// Helper for Pearson correlation for f64 vectors.
fn pearson_correlation_f64_single(vec_a: &ArrayView1<f64>, vec_b: &ArrayView1<f64>) -> Option<f64> {
    let n = vec_a.len();
    if n != vec_b.len() || n < 2 {
        return None;
    }

    let mean_a = vec_a.mean().unwrap_or(0.0);
    let mean_b = vec_b.mean().unwrap_or(0.0);

    let mut cov_ab = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let diff_a = vec_a[i] - mean_a;
        let diff_b = vec_b[i] - mean_b;
        cov_ab += diff_a * diff_b;
        var_a += diff_a * diff_a;
        var_b += diff_b * diff_b;
    }

    if var_a < 1e-12 || var_b < 1e-12 {
        // If one vector is constant (variance is ~0)
        if var_a < 1e-12 && var_b < 1e-12 { // Both constant
            // Check if they are the "same" constant. If means are very close.
             if (mean_a - mean_b).abs() < 1e-9 { return Some(1.0); } else { return Some(0.0); } // Or handle as undefined (None)
        }
        return Some(0.0); // One constant, other varies - no linear correlation
    }
    
    let r = cov_ab / (var_a.sqrt() * var_b.sqrt());
    Some(r.clamp(-1.0, 1.0))
}

/// Computes absolute Pearson correlations column-wise between m1 (f32) and m2_f64 (f64).
pub fn compute_matrix_column_correlations_abs(
    m1: &ArrayView2<f32>,
    m2_f64: &ArrayView2<f64>,
) -> Option<Vec<f64>> {
    if m1.dim() != m2_f64.dim() { return None; }
    if m1.ncols() == 0 { return Some(Vec::new()); }
    if m1.nrows() < 2 { return None; } // Need at least 2 samples for correlation

    let num_cols = m1.ncols();
    let mut correlations = Vec::with_capacity(num_cols);
    let m1_f64 = m1.mapv(|x| x as f64);

    for j in 0..num_cols {
        let col_a = m1_f64.column(j);
        let col_b = m2_f64.column(j);
        match pearson_correlation_f64_single(&col_a, &col_b) {
            Some(corr) => correlations.push(corr.abs()),
            None => return None, // Error in one of the column correlations
        }
    }
    Some(correlations)
}

/// Samples singular values, taking `count` values evenly spaced. Includes first and last if possible.
pub fn sample_singular_values(s_values: &ArrayView1<f32>, count: usize) -> Option<Vec<f32>> {
    if s_values.is_empty() || count == 0 {
        return Some(Vec::new());
    }
    if count >= s_values.len() {
        return Some(s_values.to_vec());
    }

    let mut sampled = Vec::with_capacity(count);
    let len = s_values.len();
    
    // Add the first element
    sampled.push(s_values[0]);
    if count == 1 { return Some(sampled); }

    // Calculate step for intermediate points
    // We need to pick `count - 2` more points from `len - 2` available intermediate points.
    // The indices to pick from are 1 to len-2.
    let step = (len - 2) as f64 / (count - 1) as f64; // step between selected original indices, including ends

    for i in 1..(count - 1) {
        let original_idx = (i as f64 * step).round() as usize;
        // Ensure index is within bounds [1, len-2] for intermediate values
        // This logic is simplified: take evenly spaced points across the whole array, then pick.
        // A better approach:
        // Calculate which original indices to pick based on `count` and `len`
        // Example: len=10, count=4. Pick indices 0, 3, 6, 9
        // step = (len - 1) / (count - 1)
        let pick_idx_float = i as f64 * (len - 1) as f64 / (count - 1) as f64;
        sampled.push(s_values[pick_idx_float.round() as usize]);
    }
    
    // Add the last element
    sampled.push(s_values[len - 1]);
    sampled.dedup_by(|a, b| (*a - *b).abs() < 1e-7); // Adjusted epsilon for f32
    
    Some(sampled)
}


/// Samples singular values (f64 version), taking `count` values evenly spaced. Includes first and last.
pub fn sample_singular_values_f64(s_values: &ArrayView1<f64>, count: usize) -> Option<Vec<f64>> {
    if s_values.is_empty() || count == 0 {
        return Some(Vec::new());
    }
    if count >= s_values.len() {
        return Some(s_values.to_vec());
    }

    let mut sampled = Vec::with_capacity(count);
    let len = s_values.len();

    // Always include the first singular value
    sampled.push(s_values[0]);
    if count == 1 { return Some(sampled); }

    // Determine indices for the remaining `count - 1` values
    // These should be spread across the remaining `len - 1` values
    // Example: len=10, count=4. Values from s_values[0], s_values[3], s_values[6], s_values[9]
    // Step for picking: (len - 1) / (count - 1)
    // For i from 1 to count-1: index = round(i * step)
    
    let step = (len - 1) as f64 / (count - 1) as f64;
    for i in 1..count {
        let pick_idx = (i as f64 * step).round() as usize;
        // Ensure pick_idx is within bounds and we don't re-add if rounding is weird for the last element
        // For the last iteration (i = count - 1), pick_idx should be len - 1
        if i == count - 1 {
            if sampled.last() != Some(&s_values[len-1]) { // Avoid duplicate if count=len
                 sampled.push(s_values[len - 1]);
            } else if sampled.len() < count && pick_idx == len -1 && s_values[len-1] != sampled.last().cloned().unwrap_or(f64::NAN) {
                 sampled.push(s_values[len - 1]);
            }

        } else {
             // Check if this element is different from the last one added to avoid duplicates from rounding
            if pick_idx < len -1 && (sampled.len() == 0 || s_values[pick_idx] != sampled.last().cloned().unwrap_or(f64::NAN) ) {
                 sampled.push(s_values[pick_idx]);
            } else if pick_idx < len -1 && sampled.len() < count { // If same, try to take next if not already taken
                 if pick_idx + 1 < len -1 && s_values[pick_idx+1] != sampled.last().cloned().unwrap_or(f64::NAN) {
                    sampled.push(s_values[pick_idx+1]);
                 }
            }
        }
    }
    // If due to rounding, we don't have enough, fill from end or distinct values?
    // The current logic aims for distinct points. If rounding causes fewer than `count` unique points,
    // it might return fewer. This might be acceptable.
    // A simpler strategy:
    // sampled.push(s_values[0]);
    // for i in 1..(count - 1) { sampled.push(s_values[ (i * (len -1) / (count-1)) as usize]); }
    // if count > 1 { sampled.push(s_values[len-1]); }
    // sampled.dedup(); // if strict unique values are needed and order matters less for duplicates.

    // Re-implementing sampling logic for clarity and correctness:
    // This initial block of code for f64 was a bit messy with multiple return paths.
    // The logic below is cleaner.
    // if count == 1 && len > 0 { return Some(vec![s_values[0]]); } // Covered by general logic if count=1
    // if count >= len { return Some(s_values.to_vec()); } // Covered by general logic

    let len = s_values.len(); // Original length
    if count == 0 || len == 0 { return Some(Vec::new());}
    if count >= len { return Some(s_values.to_vec()); } // If asking for more or equal to available, return all

    let mut final_sampled = Vec::with_capacity(count);
    final_sampled.push(s_values[0]); // First value

    if count > 1 {
        for i in 1..(count - 1) {
            // Calculate the index to pick for intermediate values
            // This formula spreads `count-2` points over `len-2` available intermediate slots (excluding first and last)
            // The index should be relative to the original `s_values` array.
            // Map i (from 1 to count-2) to an index in s_values (from 1 to len-2)
            let idx_float = i as f64 * (len - 1) as f64 / (count - 1) as f64;
            let idx = idx_float.round() as usize;
            final_sampled.push(s_values[idx.min(len - 1).max(0)]); // Clamp index to be safe
        }
        final_sampled.push(s_values[len - 1]); // Last value
    }
    
    final_sampled.dedup_by(|a, b| (*a - *b).abs() < 1e-9); // Keep unique values, f64 comparison

    // If dedup resulted in fewer than count, it means some values were very close.
    // This is usually fine. If a strict `count` is needed even with duplicates, remove dedup.
    Some(final_sampled)
}
