// src/diagnostics.rs

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, Ix2};
use crate::linalg_backends::{LinAlgBackendProvider, SVDOutput}; // Assuming LinAlgBackend trait itself is not directly used by helpers here
use serde::{Serialize, Deserialize};
use std::f64::INFINITY;
use std::fmt;

// --- Struct Definitions ---

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct RsvdStepDiagnostics {
    pub step_name: String,
    pub input_matrix_dims: Option<(usize, usize)>,
    pub output_matrix_dims: Option<(usize, usize)>,
    pub condition_number_input: Option<f64>,    // Optional: computed if applicable
    pub condition_number_output: Option<f64>,   // Optional: computed if applicable
    pub orthogonality_error_q: Option<f64>,     // Optional: for Q factors
    pub svd_reconstruction_error: Option<f64>,  // Optional: for SVD steps
    pub notes: String,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct PcaStepDiagnostics {
    pub stage_name: String,
    pub rsvd_diagnostics: Vec<RsvdStepDiagnostics>,
    pub u_correlation_vs_f64_truth: Option<Vec<f64>>,       // For local basis U_p vs U_p_f64_truth
    pub initial_scores_correlation_vs_py_truth: Option<Vec<f64>>, // For U_scores_star vs Python truth
    pub notes: String,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SrPassDiagnostics {
    pub pass_num: usize,
    pub v_qr_ortho_error: Option<f64>,
    pub s_intermediate_fro_norm: Option<f64>,
    pub s_intermediate_condition_number: Option<f64>,
    pub s_intermediate_svd_reconstruction_error: Option<f64>,
    pub notes: String,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct FullPcaRunDiagnostics {
    pub local_basis_diagnostics: Vec<PcaStepDiagnostics>, // One per LD block group processed
    pub condensed_matrix_fro_norm: Option<f32>,
    pub standardized_condensed_matrix_fro_norm: Option<f32>,
    pub global_pca_diagnostics: Option<Box<PcaStepDiagnostics>>, // Boxed because PcaStepDiagnostics can be large
    pub sr_pass_diagnostics: Vec<SrPassDiagnostics>,
    pub notes: String,
}

// --- Helper Function Error Type ---
// Using String for simplicity, could be a custom error enum
// #[derive(Debug, Clone)]
// pub struct DiagnosticError(String);

// impl fmt::Display for DiagnosticError {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "Diagnostic computation error: {}", self.0)
//     }
// }
// impl std::error::Error for DiagnosticError {}


// --- Helper Function Implementations ---

/// Computes the condition number of a matrix.
/// Condition number = sigma_max / sigma_min.
pub fn compute_condition_number(matrix: &ArrayView2<f32>) -> Result<f64, String> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err("Matrix is empty".to_string());
    }
    if matrix.dim().0 < matrix.dim().1 {
         // Warn or note if M < N, as SVD might be on A.T or interpretation differs.
         // For now, proceed, assuming standard interpretation for given matrix.
         // log::warn!("Condition number computed for matrix with nrows < ncols ({} < {})", matrix.nrows(), matrix.ncols());
    }

    let matrix_f64 = matrix.mapv(|x| x as f64);
    
    let backend_f64 = LinAlgBackendProvider::<f64>::new();
    let singular_values = backend_f64.svd_s(matrix_f64.into_owned(), false /* U not needed */, false /* VT not needed */)
        .map_err(|e| format!("SVD for singular values failed: {}", e))?.s;

    if singular_values.is_empty() {
        return Ok(0.0); // Or Err("No singular values computed".to_string())
    }

    let sigma_max = singular_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sigma_min_non_zero = singular_values.iter().cloned().filter(|&s| s > 1e-12).fold(f64::INFINITY, f64::min);

    if sigma_min_non_zero == f64::INFINITY || sigma_min_non_zero <= 1e-12 { // Essentially zero or no non-zero sigma_min
        return Ok(INFINITY);
    }
    if sigma_max == f64::NEG_INFINITY { // Should not happen if singular_values is not empty
        return Ok(0.0); // Or error
    }
    
    Ok(sigma_max / sigma_min_non_zero)
}

/// Computes the orthogonality error of a matrix Q, defined as ||I - Q^T Q||_F.
pub fn compute_orthogonality_error(q_matrix: &ArrayView2<f32>) -> Result<f64, String> {
    if q_matrix.nrows() == 0 || q_matrix.ncols() == 0 {
        return Err("Q matrix is empty".to_string());
    }

    let q_f64 = q_matrix.mapv(|x| x as f64);
    let qtq = q_f64.t().dot(&q_f64);
    
    let identity = Array2::<f64>::eye(qtq.nrows());
    let diff = identity - qtq;
    
    let frobenius_norm_diff = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
    Ok(frobenius_norm_diff)
}

/// Computes the SVD reconstruction error: ||A - U S V^T||_F / ||A||_F.
pub fn compute_svd_reconstruction_error(
    original_matrix: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    s_vec: &ArrayView1<f32>,
    vt: &ArrayView2<f32>,
) -> Result<f64, String> {
    if original_matrix.nrows() == 0 || original_matrix.ncols() == 0 {
        return Err("Original matrix is empty".to_string());
    }
    if u.ncols() != s_vec.len() || s_vec.len() != vt.nrows() {
        return Err(format!(
            "Dimension mismatch in SVD components: U_cols={}, S_len={}, VT_rows={}",
            u.ncols(), s_vec.len(), vt.nrows()
        ));
    }
    if u.nrows() != original_matrix.nrows() || vt.ncols() != original_matrix.ncols() {
         return Err(format!(
            "Dimension mismatch between original matrix and SVD components: Orig_rows={}, U_rows={}; Orig_cols={}, VT_cols={}",
            original_matrix.nrows(), u.nrows(), original_matrix.ncols(), vt.ncols()
        ));
    }

    let original_f64 = original_matrix.mapv(|x| x as f64);
    let u_f64 = u.mapv(|x| x as f64);
    let s_diag_f64 = Array2::from_diag(&s_vec.mapv(|x| x as f64));
    let vt_f64 = vt.mapv(|x| x as f64);

    // A_reco = U * S_diag * VT
    let u_s = u_f64.dot(&s_diag_f64);
    let reconstructed_f64 = u_s.dot(&vt_f64);

    let diff = original_f64.clone() - reconstructed_f64;
    
    let norm_diff = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_original = original_f64.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if norm_original < 1e-12 { // original matrix is close to zero
        if norm_diff < 1e-12 { // diff is also close to zero
            return Ok(0.0); // Perfect reconstruction of a zero matrix
        } else {
            // Non-zero difference from a zero matrix, effectively infinite relative error
            // Or could return norm_diff itself if that's more informative
            return Ok(INFINITY); 
        }
    }
    Ok(norm_diff / norm_original)
}

/// Helper for Pearson correlation.
fn pearson_correlation_f64(vec_a: &ArrayView1<f64>, vec_b: &ArrayView1<f64>) -> Result<f64, String> {
    let n = vec_a.len();
    if n != vec_b.len() {
        return Err("Vectors have different lengths".to_string());
    }
    if n < 2 {
        return Err("Vectors are too short to compute correlation (min length 2)".to_string());
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

    // Using sample standard deviation (/(n-1)) implicitly by not dividing cov_ab, var_a, var_b by n or n-1 consistently
    // The n or n-1 factor cancels out in the ratio for r.
    // However, ensure no division by zero for std dev.
    if var_a < 1e-12 || var_b < 1e-12 {
        // If one vector is constant, correlation is undefined or zero depending on convention.
        // Let's return 0 if one var is zero, implying no linear relationship can be measured.
        // If both are constant and means match, could be 1.0. If means differ, could be NaN.
        // For simplicity, if either variance is zero, return 0.0.
        // A more robust solution might return NaN or specific error.
        if var_a < 1e-12 && var_b < 1e-12 { // both constant
            // If both are constant, their correlation is typically taken as undefined (NaN) or 1 if they are "identical" constants (which they are).
            // Let's return 1.0 if both are constant. Or 0.0 if we want to signal no *varying* relationship.
            // For now, returning 0.0 for any zero variance.
            return Ok(0.0); 
        }
        return Ok(0.0); 
    }
    
    let r = cov_ab / (var_a.sqrt() * var_b.sqrt());
    Ok(r.clamp(-1.0, 1.0)) // Clamp to handle potential floating point inaccuracies
}


/// Computes Pearson correlation between a f32 vector and a f64 vector.
pub fn compute_vector_correlation(vec_a_f32: &ArrayView1<f32>, vec_b_f64: &ArrayView1<f64>) -> Result<f64, String> {
    let vec_a_f64 = vec_a_f32.mapv(|x| x as f64);
    pearson_correlation_f64(&vec_a_f64.view(), vec_b_f64)
}

/// Computes column-wise Pearson correlations between two matrices (f32 and f64).
pub fn compute_matrix_correlations(
    matrix_a_f32: &ArrayView2<f32>,
    matrix_b_f64: &ArrayView2<f64>,
) -> Result<Vec<f64>, String> {
    if matrix_a_f32.dim() != matrix_b_f64.dim() {
        return Err(format!(
            "Matrices have different dimensions: A({:?}), B({:?})",
            matrix_a_f32.dim(), matrix_b_f64.dim()
        ));
    }
    if matrix_a_f32.ncols() == 0 {
        return Ok(Vec::new()); // No columns to correlate
    }
    if matrix_a_f32.nrows() < 2 {
        return Err("Matrices have too few rows (<2) to compute column correlations".to_string());
    }


    let num_cols = matrix_a_f32.ncols();
    let mut correlations = Vec::with_capacity(num_cols);

    let matrix_a_f64 = matrix_a_f32.mapv(|x| x as f64);

    for j in 0..num_cols {
        let col_a = matrix_a_f64.column(j);
        let col_b = matrix_b_f64.column(j);
        match pearson_correlation_f64(&col_a, &col_b) {
            Ok(corr) => correlations.push(corr),
            Err(e) => return Err(format!("Failed to compute correlation for column {}: {}", j, e)),
        }
    }
    Ok(correlations)
}

// Placeholder for LinAlgBackend trait if it was meant to be used directly
// pub use crate::linalg_backends::LinAlgBackend;
// Need to ensure `LinAlgBackendProvider` can be created for f64.
// The `svd_s` method is assumed to exist on `LinAlgBackendProvider<f64>`.
// If not, this would need adjustment based on the actual backend trait structure.
// For example, if `LinAlgBackend` is the trait, then:
// `let backend_f64 = NdarrayLinAlgBackend::new(); // Or whatever concrete f64 backend
// let singular_values = backend_f64.svd_s_f64(...)`
// But current structure seems to be `LinAlgBackendProvider::<F>::new().svd_s(...)` which is fine.

// Note on SVDOutput:
// The `SVDOutput` struct is defined in `linalg_backends` (based on previous files).
// It typically has fields like `u: Option<Array2<F>>`, `s: Array1<F::Real>`, `vt: Option<Array2<F>>`.
// For `svd_s` used in `compute_condition_number`, we are only interested in `s`.
// The `LinAlgBackendProvider`'s `svd_s` method should return an `SVDOutput` where only `s` is populated.
// (Or a more direct method like `singular_values()` if available).
// The current `svd_s` call `backend_f64.svd_s(matrix_f64.into_owned(), false, false)` seems to align with this.
// The `SVDOutput` struct itself is not directly used here beyond what `svd_s` returns and deconstructs.
// The import `use crate::linalg_backends::{LinAlgBackendProvider, SVDOutput};` is mainly for `LinAlgBackendProvider`.
// `SVDOutput` might not be strictly necessary for this file if `svd_s` result is directly `s`.
// However, if `svd_s` returns `Result<SVDOutput<f64>, Error>`, then accessing `.s` is correct.
// I'll keep the `SVDOutput` import for now as it doesn't harm and reflects potential structure.

```
