// src/linalg_backends.rs

#[derive(Debug, Default, Copy, Clone)]
pub struct LinAlgBackendProvider<F: 'static + Copy + Send + Sync> {
    _phantom: PhantomData<F>,
}

impl<F: 'static + Copy + Send + Sync> LinAlgBackendProvider<F> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

// --- Common imports needed by multiple sections ---
use ndarray::{Array1, Array2};
use std::error::Error;
use std::marker::PhantomData;

// --- Trait Definitions (originally from linalg_abstract.rs) ---

/// Output of a symmetric eigendecomposition.
#[derive(Debug)]
pub struct EighOutput<F: 'static> {
    /// Eigenvalues, typically sorted in ascending order.
    pub eigenvalues: Array1<F>,
    /// Eigenvectors as columns of the matrix.
    /// eigenvector_matrix.column(i) corresponds to eigenvalues[i].
    pub eigenvectors: Array2<F>,
}

/// Trait for symmetric eigendecomposition (similar to LAPACK's DSYEVR or DSYEVD).
/// Implementers will typically expect `matrix` to be symmetric.
pub trait BackendEigh<F: 'static + Copy + Send + Sync> {
    fn eigh_upper(&self, matrix: &Array2<F>) -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>>;
}

/// Trait for QR decomposition, focusing on retrieving the Q factor.
pub trait BackendQR<F: 'static + Copy + Send + Sync> {
    fn qr_q_factor(&self, matrix: &Array2<F>) -> Result<Array2<F>, Box<dyn Error + Send + Sync>>;
}

/// Output of a Singular Value Decomposition.
#[derive(Debug)]
pub struct SVDOutput<F: 'static> {
    pub u: Option<Array2<F>>,
    pub s: Array1<F>,
    pub vt: Option<Array2<F>>,
}

/// Trait for Singular Value Decomposition.
pub trait BackendSVD<F: 'static + Copy + Send + Sync> {
    fn svd_into(&self, matrix: Array2<F>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>>;
}

// --- NdarrayLinAlgBackend Implementation (originally from ndarray_backend.rs) ---
// Specific imports for ndarray-linalg backend
// use ndarray::ScalarOperand; // Removed as not directly used by trait impls
use ndarray_linalg::{Eigh as NdLinalgEigh, QR as NdLinalgQR, SVDInto as NdLinalgSVDInto, UPLO};
// use num_traits::AsPrimitive; // Removed as not directly used by trait impls

// Define a concrete type for ndarray-linalg backend
#[derive(Debug, Default, Copy, Clone)]
pub struct NdarrayLinAlgBackend;

// Helper to convert ndarray-linalg's error to Box<dyn Error + Send + Sync>
fn to_dyn_error<E: Error + Send + Sync + 'static>(e: E) -> Box<dyn Error + Send + Sync> {
    Box::new(e)
}

// --- Implementations for f64 ---
impl BackendEigh<f64> for NdarrayLinAlgBackend {
    fn eigh_upper(&self, matrix: &Array2<f64>) -> Result<EighOutput<f64>, Box<dyn Error + Send + Sync>> {
        let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Upper).map_err(to_dyn_error)?;
        Ok(EighOutput { eigenvalues, eigenvectors })
    }
}

impl BackendQR<f64> for NdarrayLinAlgBackend {
    fn qr_q_factor(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error + Send + Sync>> {
        let (q_factor, _r) = matrix.qr().map_err(to_dyn_error)?;
        Ok(q_factor)
    }
}

impl BackendSVD<f64> for NdarrayLinAlgBackend {
    fn svd_into(&self, matrix: Array2<f64>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<f64>, Box<dyn Error + Send + Sync>> {
        let (u, s, vt) = matrix.svd_into(compute_u, compute_v).map_err(to_dyn_error)?;
        Ok(SVDOutput { u, s, vt })
    }
}

// --- Implementations for f32 ---
impl BackendEigh<f32> for NdarrayLinAlgBackend {
    fn eigh_upper(&self, matrix: &Array2<f32>) -> Result<EighOutput<f32>, Box<dyn Error + Send + Sync>> {
        let (eigenvalues, eigenvectors) = matrix.eigh(UPLO::Upper).map_err(to_dyn_error)?;
        Ok(EighOutput { eigenvalues, eigenvectors })
    }
}

impl BackendQR<f32> for NdarrayLinAlgBackend {
    fn qr_q_factor(&self, matrix: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        let (q_factor, _r) = matrix.qr().map_err(to_dyn_error)?;
        Ok(q_factor)
    }
}

impl BackendSVD<f32> for NdarrayLinAlgBackend {
    fn svd_into(&self, matrix: Array2<f32>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<f32>, Box<dyn Error + Send + Sync>> {
        let (u, s, vt) = matrix.svd_into(compute_u, compute_v).map_err(to_dyn_error)?;
        Ok(SVDOutput { u, s, vt })
    }
}


// --- FaerLinAlgBackend Implementation (originally from faer_backend.rs) ---
#[cfg(feature = "backend_faer")]
mod faer_specific_code { // Encapsulate faer-specific code and its imports
    use super::{BackendEigh, BackendQR, BackendSVD, EighOutput, SVDOutput};
    use ndarray::{Array1, Array2, ShapeBuilder};
    use faer::{MatRef}; // Keep MatRef, other specific imports will be added
    use faer::traits::num_traits::Zero;     // Use Zero via faer's re-export
    use faer::traits::ComplexField;
    use bytemuck::Pod;
    use std::error::Error; // Should already be there from original code
    
    // Updated imports for SVD
    // use faer::Parallelism; // No longer needed
    // use faer::dyn_stack::GlobalPodBuffer; // No longer needed
    // use faer::linalg::svd::ComputeSvdVectors as ComputeVectors; // Commented out as likely not needed
    use faer::linalg::solvers::Svd as FaerSolverSvd; // Alias for the new SVD solver
    // SvdReq is likely not needed.


    fn to_dyn_error_faer(msg: String) -> Box<dyn Error + Send + Sync> {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, msg))
    }

    #[derive(Debug, Default, Copy, Clone)]
    pub struct FaerLinAlgBackend;

    fn faer_mat_to_ndarray<F: ComplexField + Copy + Pod + Zero>(faer_mat: MatRef<'_, F>) -> Array2<F> {
        let nrows = faer_mat.nrows();
        let ncols = faer_mat.ncols();
        if nrows == 0 || ncols == 0 {
            return Array2::zeros((nrows, ncols).f());
        }
        let mut data_vec = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data_vec.push(unsafe { *faer_mat.get_unchecked(i, j) });
            }
        }
        Array2::from_shape_vec((nrows, ncols).f(), data_vec)
            .expect("Shape and data length mismatch creating ndarray from faer Mat")
    }

    fn faer_col_to_ndarray_vec<F: ComplexField + Copy + Pod + Zero>(faer_col: faer::ColRef<'_, F>) -> Array1<F> {
        let nrows = faer_col.nrows();
        if nrows == 0 {
            return Array1::zeros(0);
        }
        let mut data_vec = Vec::with_capacity(nrows);
        for i in 0..nrows {
            data_vec.push(unsafe { *faer_col.get_unchecked(i) });
        }
        Array1::from_vec(data_vec)
    }

    impl BackendEigh<f64> for FaerLinAlgBackend {
        fn eigh_upper(&self, matrix: &Array2<f64>) -> Result<EighOutput<f64>, Box<dyn Error + Send + Sync>> {
            if matrix.nrows() != matrix.ncols() {
                return Err(to_dyn_error_faer("Matrix must be square for eigendecomposition.".to_string()));
            }
            if matrix.is_empty() {
                return Ok(EighOutput { eigenvalues: Array1::zeros(0), eigenvectors: Array2::zeros((0,0)) });
            }
            let matrix_view = matrix.view(); // matrix is &Array2<F> or Array2<F>
            let nrows = matrix_view.nrows();
            let ncols = matrix_view.ncols();

            let faer_mat_view = if matrix_view.is_standard_layout() { // C-order (row-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // This case should ideally not happen if is_standard_layout() is true,
                    // as standard layout implies C-contiguous.
                    // However, if it can, a copy would be needed.
                    // For now, error if slice is None, as direct view is not possible.
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from row-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else if matrix_view.is_fcontiguous() { // Fortran-order (column-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                } else {
                    // Similar to above, this case implies non-contiguous F-layout, which is unusual.
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from column-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else {
                // Non-contiguous and not simply C or F layout.
                // faer::MatRef requires contiguous data.
                // A possible fallback would be to create an owned faer::Mat by copying,
                // then take a view, but this changes the function's nature (no longer just viewing).
                // For now, error out as a direct view isn't possible.
                return Err(to_dyn_error_faer(format!(
                    "Input ndarray matrix ({}x{}) is non-contiguous and cannot be directly viewed by faer. Consider making a contiguous copy.", nrows, ncols
                )));
            };
            let eig = faer_mat_view.as_ref().selfadjoint_eigendecomposition(faer::Side::Upper);
            let eigenvalues_faer_colref = eig.s();
            let eigenvectors_faer_matref = eig.u();
            Ok(EighOutput {
                eigenvalues: faer_col_to_ndarray_vec(eigenvalues_faer_colref.as_ref()),
                eigenvectors: faer_mat_to_ndarray(eigenvectors_faer_matref.as_ref()),
            })
        }
    }

    impl BackendQR<f64> for FaerLinAlgBackend {
        fn qr_q_factor(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            let k = nrows.min(ncols);
            if nrows == 0 { return Ok(Array2::zeros((0, k))); }
            let matrix_view = matrix.view(); // matrix is &Array2<F> or Array2<F>
            // let nrows = matrix_view.nrows(); // already available
            // let ncols = matrix_view.ncols(); // already available

            let faer_mat_view = if matrix_view.is_standard_layout() { // C-order (row-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from row-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else if matrix_view.is_fcontiguous() { // Fortran-order (column-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from column-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else {
                return Err(to_dyn_error_faer(format!(
                    "Input ndarray matrix ({}x{}) is non-contiguous and cannot be directly viewed by faer. Consider making a contiguous copy.", nrows, ncols
                )));
            };
            let qr_decomp = faer_mat_view.as_ref().qr(); // This is faer::MatRef::qr() which returns faer::linalg::solvers::Qr
            let q_thin_faer_mat = qr_decomp.compute_thin_q(); // This returns an owned Mat<T>
            Ok(faer_mat_to_ndarray(q_thin_faer_mat.as_ref())) // Pass as MatRef
        }
    }

    impl BackendSVD<f64> for FaerLinAlgBackend {
        fn svd_into(&self, matrix: Array2<f64>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<f64>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if matrix.is_empty() {
                let k_dim = nrows.min(ncols);
                 return Ok(SVDOutput {
                    u: if compute_u { Some(Array2::zeros((nrows, k_dim))) } else { None },
                    s: Array1::zeros(k_dim),
                    vt: if compute_v { Some(Array2::zeros((k_dim, ncols))) } else { None },
                });
            }
            let matrix_view = matrix.view(); // matrix is Array2<F>

            let faer_mat_view = if matrix_view.is_standard_layout() { // C-order (row-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from row-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else if matrix_view.is_fcontiguous() { // Fortran-order (column-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from column-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else {
                return Err(to_dyn_error_faer(format!(
                    "Input ndarray matrix ({}x{}) is non-contiguous and cannot be directly viewed by faer. Consider making a contiguous copy.", nrows, ncols
                )));
            };
            let faer_mat_ref = faer_mat_view.as_ref(); 

            let svd_solver_instance = FaerSolverSvd::new_thin(faer_mat_ref)
                .map_err(|e| to_dyn_error_faer(format!("Faer SVD computation failed: {:?}", e)))?;

            let s_diag_ref = svd_solver_instance.S(); 
            let s_ndarray = faer_col_to_ndarray_vec(s_diag_ref.diagonal().as_ref());

            let u_ndarray = if compute_u {
                Some(faer_mat_to_ndarray(svd_solver_instance.U().as_ref()))
            } else { None };

            let vt_ndarray = if compute_v {
                let v_mat_ref = svd_solver_instance.V();
                let v_ndarray = faer_mat_to_ndarray(v_mat_ref.as_ref());
                Some(v_ndarray.t().into_owned())
            } else { None };

            Ok(SVDOutput { u: u_ndarray, s: s_ndarray, vt: vt_ndarray })
        }
    }

    impl BackendEigh<f32> for FaerLinAlgBackend {
        fn eigh_upper(&self, matrix: &Array2<f32>) -> Result<EighOutput<f32>, Box<dyn Error + Send + Sync>> {
            if matrix.nrows() != matrix.ncols() {
                return Err(to_dyn_error_faer("Matrix must be square for eigendecomposition.".to_string()));
            }
            if matrix.is_empty() {
                return Ok(EighOutput { eigenvalues: Array1::zeros(0), eigenvectors: Array2::zeros((0,0)) });
            }
            let matrix_view = matrix.view(); // matrix is &Array2<F> or Array2<F>
            let nrows = matrix_view.nrows();
            let ncols = matrix_view.ncols();

            let faer_mat_view = if matrix_view.is_standard_layout() { // C-order (row-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from row-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else if matrix_view.is_fcontiguous() { // Fortran-order (column-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from column-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else {
                return Err(to_dyn_error_faer(format!(
                    "Input ndarray matrix ({}x{}) is non-contiguous and cannot be directly viewed by faer. Consider making a contiguous copy.", nrows, ncols
                )));
            };
            let eig = faer_mat_view.as_ref().selfadjoint_eigendecomposition(faer::Side::Upper);
            let eigenvalues_faer_colref = eig.s();
            let eigenvectors_faer_matref = eig.u();
            Ok(EighOutput {
                eigenvalues: faer_col_to_ndarray_vec(eigenvalues_faer_colref.as_ref()),
                eigenvectors: faer_mat_to_ndarray(eigenvectors_faer_matref.as_ref()),
            })
        }
    }

    impl BackendQR<f32> for FaerLinAlgBackend {
        fn qr_q_factor(&self, matrix: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            let k = nrows.min(ncols);
            if nrows == 0 { return Ok(Array2::zeros((0, k))); }
            let matrix_view = matrix.view(); // matrix is &Array2<F> or Array2<F>
            // let nrows = matrix_view.nrows(); // already available
            // let ncols = matrix_view.ncols(); // already available

            let faer_mat_view = if matrix_view.is_standard_layout() { // C-order (row-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from row-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else if matrix_view.is_fcontiguous() { // Fortran-order (column-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from column-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else {
                return Err(to_dyn_error_faer(format!(
                    "Input ndarray matrix ({}x{}) is non-contiguous and cannot be directly viewed by faer. Consider making a contiguous copy.", nrows, ncols
                )));
            };
            let qr_decomp = faer_mat_view.as_ref().qr(); // This is faer::MatRef::qr() which returns faer::linalg::solvers::Qr
            let q_thin_faer_mat = qr_decomp.compute_thin_q(); // This returns an owned Mat<T>
            Ok(faer_mat_to_ndarray(q_thin_faer_mat.as_ref())) // Pass as MatRef
        }
    }

    impl BackendSVD<f32> for FaerLinAlgBackend {
        fn svd_into(&self, matrix: Array2<f32>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<f32>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if matrix.is_empty() {
                let k_dim = nrows.min(ncols);
                 return Ok(SVDOutput {
                    u: if compute_u { Some(Array2::zeros((nrows, k_dim))) } else { None },
                    s: Array1::zeros(k_dim),
                    vt: if compute_v { Some(Array2::zeros((k_dim, ncols))) } else { None },
                });
            }
            let matrix_view = matrix.view(); // matrix is Array2<F>

            let faer_mat_view = if matrix_view.is_standard_layout() { // C-order (row-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from row-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else if matrix_view.is_fcontiguous() { // Fortran-order (column-major)
                if let Some(slice) = matrix_view.as_slice_memory_order() {
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                } else {
                    return Err(to_dyn_error_faer(format!(
                        "Failed to get slice from column-major ndarray matrix ({}x{})", nrows, ncols
                    )));
                }
            } else {
                return Err(to_dyn_error_faer(format!(
                    "Input ndarray matrix ({}x{}) is non-contiguous and cannot be directly viewed by faer. Consider making a contiguous copy.", nrows, ncols
                )));
            };
            let faer_mat_ref = faer_mat_view.as_ref();

            let svd_solver_instance = FaerSolverSvd::new_thin(faer_mat_ref)
                .map_err(|e| to_dyn_error_faer(format!("Faer SVD computation failed: {:?}", e)))?;

            let s_diag_ref = svd_solver_instance.S();
            let s_ndarray = faer_col_to_ndarray_vec(s_diag_ref.diagonal().as_ref());

            let u_ndarray = if compute_u {
                Some(faer_mat_to_ndarray(svd_solver_instance.U().as_ref()))
            } else { None };

            let vt_ndarray = if compute_v {
                let v_mat_ref = svd_solver_instance.V();
                let v_ndarray = faer_mat_to_ndarray(v_mat_ref.as_ref());
                Some(v_ndarray.t().into_owned())
            } else { None };

            Ok(SVDOutput { u: u_ndarray, s: s_ndarray, vt: vt_ndarray })
        }
    }
} // End of faer_specific_code module

// --- LinAlgBackendProvider Dispatch (originally from linalg_backend_dispatch.rs) ---

// Import concrete backend types for the provider
#[cfg(feature = "backend_faer")]
// // use self::faer_specific_code::FaerLinAlgBackend; // Path adjusted to inner module - This line is removed

// NdarrayLinAlgBackend is already defined in this file.

/// A provider struct that dispatches to the selected linear algebra backend
/// based on compile-time feature flags.

// --- Implement BackendEigh for Provider ---
impl<F> BackendEigh<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendEigh<F>, // NdarrayLinAlgBackend is in the same file (self)
    #[cfg(feature = "backend_faer")]
    faer_specific_code::FaerLinAlgBackend: BackendEigh<F>, // FaerLinAlgBackend is in inner module
{
    fn eigh_upper(&self, matrix: &Array2<F>) -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>> {
        #[cfg(feature = "backend_faer")]
        {
            faer_specific_code::FaerLinAlgBackend.eigh_upper(matrix)
        }
        #[cfg(not(feature = "backend_faer"))]
        {
            NdarrayLinAlgBackend.eigh_upper(matrix)
        }
    }
}

// --- Implement BackendQR for Provider ---
impl<F> BackendQR<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendQR<F>,
    #[cfg(feature = "backend_faer")]
    faer_specific_code::FaerLinAlgBackend: BackendQR<F>,
{
    fn qr_q_factor(&self, matrix: &Array2<F>) -> Result<Array2<F>, Box<dyn Error + Send + Sync>> {
        #[cfg(feature = "backend_faer")]
        {
            faer_specific_code::FaerLinAlgBackend.qr_q_factor(matrix)
        }
        #[cfg(not(feature = "backend_faer"))]
        {
            NdarrayLinAlgBackend.qr_q_factor(matrix)
        }
    }
}

// --- Implement BackendSVD for Provider ---
impl<F> BackendSVD<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendSVD<F>,
    #[cfg(feature = "backend_faer")]
    faer_specific_code::FaerLinAlgBackend: BackendSVD<F>,
{
    fn svd_into(&self, matrix: Array2<F>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>> {
        #[cfg(feature = "backend_faer")]
        {
            faer_specific_code::FaerLinAlgBackend.svd_into(matrix, compute_u, compute_v)
        }
        #[cfg(not(feature = "backend_faer"))]
        {
            NdarrayLinAlgBackend.svd_into(matrix, compute_u, compute_v)
        }
    }
}
