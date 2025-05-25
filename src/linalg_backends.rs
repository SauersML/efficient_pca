// src/linalg_backends.rs

// --- Common imports needed by multiple sections ---
use ndarray::{Array1, Array2};
use std::error::Error;

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
    use super::{BackendEigh, BackendQR, BackendSVD, EighOutput, SVDOutput}; // Use traits from parent module
    use ndarray::{Array1, Array2, ArrayView, ArrayViewMut, ShapeBuilder}; 
    use faer::linalg::svd::{ComputeSvdVectors, SvdParams};
    use faer::linalg::householder::Reconstruct; // For QR's q().reconstruct()
    use faer::{MatRef, MatMut, Par}; 
    use faer::diag::DiagMut; // Specific path for DiagMut type alias
    use faer::dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
    use faer::traits::Number; // Added import for Number trait
    use std::error::Error;

    fn to_dyn_error_faer(msg: String) -> Box<dyn Error + Send + Sync> {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, msg))
    }

    #[derive(Debug, Default, Copy, Clone)]
    pub struct FaerLinAlgBackend;

    fn faer_mat_to_ndarray<F: Number>(faer_mat: MatRef<'_, F>) -> Array2<F> {
        let nrows = faer_mat.nrows();
        let ncols = faer_mat.ncols();
        if nrows == 0 || ncols == 0 {
            return Array2::zeros((nrows, ncols).f());
        }
        let mut data_vec = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data_vec.push(faer_mat.read(i, j));
            }
        }
        Array2::from_shape_vec((nrows, ncols).f(), data_vec)
            .expect("Shape and data length mismatch creating ndarray from faer Mat")
    }

    fn faer_col_to_ndarray_vec<F: Number>(faer_col: faer::ColRef<'_, F>) -> Array1<F> {
        let nrows = faer_col.nrows();
        if nrows == 0 {
            return Array1::zeros(0);
        }
        let mut data_vec = Vec::with_capacity(nrows);
        for i in 0..nrows {
            data_vec.push(faer_col.read(i));
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
            let matrix_view = matrix.view();
            let faer_mat_view = unsafe {
                MatRef::from_raw_parts(
                    matrix_view.as_ptr(),
                    matrix_view.nrows(),
                    matrix_view.ncols(),
                    matrix_view.strides()[0] as isize,
                    matrix_view.strides()[1] as isize,
                )
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
            let matrix_view = matrix.view();
            let faer_mat_view = unsafe {
                MatRef::from_raw_parts(
                    matrix_view.as_ptr(),
                    matrix_view.nrows(),
                    matrix_view.ncols(),
                    matrix_view.strides()[0] as isize,
                    matrix_view.strides()[1] as isize,
                )
            };
            let qr_decomp = faer_mat_view.as_ref().qr();
            let q_full_faer_mat = qr_decomp.q().reconstruct();
            let q_thin_faer_ref = q_full_faer_mat.as_ref().submatrix(0, 0, nrows, k);
            Ok(faer_mat_to_ndarray(q_thin_faer_ref))
        }
    }

    impl BackendSVD<f64> for FaerLinAlgBackend {
        fn svd_into(&self, matrix: Array2<f64>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<f64>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            let k_dim = nrows.min(ncols);

            if matrix.is_empty() {
                 return Ok(SVDOutput {
                    u: if compute_u { Some(Array2::zeros((nrows, k_dim))) } else { None },
                    s: Array1::zeros(k_dim),
                    vt: if compute_v { Some(Array2::zeros((k_dim, ncols))) } else { None },
                });
            }

            let matrix_view = matrix.view();
            let faer_mat_view = unsafe {
                MatRef::from_raw_parts(
                    matrix_view.as_ptr(),
                    matrix_view.nrows(),
                    matrix_view.ncols(),
                    matrix_view.strides()[0] as isize,
                    matrix_view.strides()[1] as isize,
                )
            };

            // Prepare output arrays
            let mut s_values = Array1::<f64>::zeros(k_dim);
            let mut u_matrix = if compute_u { Array2::<f64>::zeros((nrows, k_dim)) } else { Array2::<f64>::zeros((0,0)) }; // Dummy if not used
            let mut v_matrix = if compute_v { Array2::<f64>::zeros((ncols, k_dim)) } else { Array2::<f64>::zeros((0,0)) }; // Dummy if not used

            // Convert to Faer mutable views
            let mut s_diag_mut = faer::diag::Diag::from_slice_mut(s_values.as_slice_mut().unwrap(), 0);
            let mut u_mat_mut_opt = if compute_u {
                Some(unsafe {
                    MatMut::from_raw_parts(
                        u_matrix.as_mut_ptr(),
                        u_matrix.nrows(),
                        u_matrix.ncols(),
                        u_matrix.strides()[0] as isize,
                        u_matrix.strides()[1] as isize,
                    )
                })
            } else { None };

            let mut v_mat_mut_opt = if compute_v {
                Some(unsafe {
                    MatMut::from_raw_parts(
                        v_matrix.as_mut_ptr(),
                        v_matrix.nrows(),
                        v_matrix.ncols(),
                        v_matrix.strides()[0] as isize,
                        v_matrix.strides()[1] as isize,
                    )
                })
            } else { None };
            
            // Memory allocation for Faer
            let compute_u_svd_enum = if compute_u { ComputeSvdVectors::Full } else { ComputeSvdVectors::None };
            let compute_v_svd_enum = if compute_v { ComputeSvdVectors::Full } else { ComputeSvdVectors::None };

            let stack_req = faer::linalg::svd::svd_scratch::<f64>(
                faer_mat_view.nrows(),
                faer_mat_view.ncols(),
                compute_u_svd_enum,
                compute_v_svd_enum,
                SvdParams::default() 
            ).map_err(|e| to_dyn_error_faer(format!("SVD scratch requirements error: {:?}", e)))?;
            
            let mut pod_buffer = GlobalPodBuffer::new(stack_req);
            let mut mem_stack = PodStack::new(&mut pod_buffer);

            // Call faer::linalg::svd::svd
            faer::linalg::svd::svd(
                faer_mat_view.as_ref(),
                &mut s_diag_mut, // Pass DiagMut by mutable reference
                u_mat_mut_opt.as_mut().map(|m| m.as_mut()), // Option<MatMut<E>>
                v_mat_mut_opt.as_mut().map(|m| m.as_mut()), // Option<MatMut<E>>
                Par::Seq,
                &mut mem_stack,
                SvdParams::default()
            ).map_err(|e| to_dyn_error_faer(format!("SVD computation failed: {:?}", e)))?;
            
            // Construct SVDOutput
            let u_ndarray = if compute_u { Some(u_matrix) } else { None };
            let vt_ndarray = if compute_v { Some(v_matrix.t().into_owned()) } else { None };

            Ok(SVDOutput { u: u_ndarray, s: s_values, vt: vt_ndarray })
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
            let matrix_view = matrix.view();
            let faer_mat_view = unsafe {
                MatRef::from_raw_parts(
                    matrix_view.as_ptr(),
                    matrix_view.nrows(),
                    matrix_view.ncols(),
                    matrix_view.strides()[0] as isize,
                    matrix_view.strides()[1] as isize,
                )
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
            let matrix_view = matrix.view();
            let faer_mat_view = unsafe {
                MatRef::from_raw_parts(
                    matrix_view.as_ptr(),
                    matrix_view.nrows(),
                    matrix_view.ncols(),
                    matrix_view.strides()[0] as isize,
                    matrix_view.strides()[1] as isize,
                )
            };
            let qr_decomp = faer_mat_view.as_ref().qr();
            let q_full_faer_mat = qr_decomp.q().reconstruct();
            let q_thin_faer_ref = q_full_faer_mat.as_ref().submatrix(0, 0, nrows, k);
            Ok(faer_mat_to_ndarray(q_thin_faer_ref))
        }
    }

    impl BackendSVD<f32> for FaerLinAlgBackend {
        fn svd_into(&self, matrix: Array2<f32>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<f32>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            let k_dim = nrows.min(ncols);

            if matrix.is_empty() {
                 return Ok(SVDOutput {
                    u: if compute_u { Some(Array2::zeros((nrows, k_dim))) } else { None },
                    s: Array1::zeros(k_dim),
                    vt: if compute_v { Some(Array2::zeros((k_dim, ncols))) } else { None },
                });
            }

            let matrix_view = matrix.view();
            let faer_mat_view = unsafe {
                MatRef::from_raw_parts(
                    matrix_view.as_ptr(),
                    matrix_view.nrows(),
                    matrix_view.ncols(),
                    matrix_view.strides()[0] as isize,
                    matrix_view.strides()[1] as isize,
                )
            };

            // Prepare output arrays
            let mut s_values = Array1::<f32>::zeros(k_dim);
            let mut u_matrix = if compute_u { Array2::<f32>::zeros((nrows, k_dim)) } else { Array2::<f32>::zeros((0,0)) };
            let mut v_matrix = if compute_v { Array2::<f32>::zeros((ncols, k_dim)) } else { Array2::<f32>::zeros((0,0)) };

            // Convert to Faer mutable views
            let mut s_diag_mut = faer::diag::Diag::from_slice_mut(s_values.as_slice_mut().unwrap(), 0);
            let mut u_mat_mut_opt = if compute_u {
                Some(unsafe {
                    MatMut::from_raw_parts(
                        u_matrix.as_mut_ptr(),
                        u_matrix.nrows(),
                        u_matrix.ncols(),
                        u_matrix.strides()[0] as isize,
                        u_matrix.strides()[1] as isize,
                    )
                })
            } else { None };

            let mut v_mat_mut_opt = if compute_v {
                Some(unsafe {
                    MatMut::from_raw_parts(
                        v_matrix.as_mut_ptr(),
                        v_matrix.nrows(),
                        v_matrix.ncols(),
                        v_matrix.strides()[0] as isize,
                        v_matrix.strides()[1] as isize,
                    )
                })
            } else { None };
            
            // Memory allocation for Faer
            let compute_u_svd_enum = if compute_u { ComputeSvdVectors::Full } else { ComputeSvdVectors::None };
            let compute_v_svd_enum = if compute_v { ComputeSvdVectors::Full } else { ComputeSvdVectors::None };

            let stack_req = faer::linalg::svd::svd_scratch::<f32>(
                faer_mat_view.nrows(),
                faer_mat_view.ncols(),
                compute_u_svd_enum,
                compute_v_svd_enum,
                SvdParams::default()
            ).map_err(|e| to_dyn_error_faer(format!("SVD f32 scratch requirements error: {:?}", e)))?;
            
            let mut pod_buffer = GlobalPodBuffer::new(stack_req);
            let mut mem_stack = PodStack::new(&mut pod_buffer);

            // Call faer::linalg::svd::svd
            faer::linalg::svd::svd(
                faer_mat_view.as_ref(),
                &mut s_diag_mut,
                u_mat_mut_opt.as_mut().map(|m| m.as_mut()),
                v_mat_mut_opt.as_mut().map(|m| m.as_mut()),
                Par::Seq,
                &mut mem_stack,
                faer::Spec::auto() // Use Spec::auto() for the params argument
            ).map_err(|e| to_dyn_error_faer(format!("SVD f32 computation failed: {:?}", e)))?;
            
            // Construct SVDOutput
            let u_ndarray = if compute_u { Some(u_matrix) } else { None };
            let vt_ndarray = if compute_v { Some(v_matrix.t().into_owned()) } else { None };

            Ok(SVDOutput { u: u_ndarray, s: s_values, vt: vt_ndarray })
        }
    }
} // End of faer_specific_code module

// --- LinAlgBackendProvider Dispatch (originally from linalg_backend_dispatch.rs) ---
use std::marker::PhantomData;

// NdarrayLinAlgBackend is already defined in this file.

/// A provider struct that dispatches to the selected linear algebra backend
/// based on compile-time feature flags.
#[derive(Debug, Default, Copy, Clone)]
pub struct LinAlgBackendProvider<F: 'static + Copy + Send + Sync> {
    _phantom: PhantomData<F>,
}

impl<F: 'static + Copy + Send + Sync> LinAlgBackendProvider<F> {
    pub fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

// --- Implement BackendEigh for Provider ---
#[cfg(feature = "backend_faer")]
impl<F> BackendEigh<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendEigh<F>, 
    faer_specific_code::FaerLinAlgBackend: BackendEigh<F>,
{
    fn eigh_upper(&self, matrix: &Array2<F>) -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.eigh_upper(matrix)
    }
}

#[cfg(not(feature = "backend_faer"))]
impl<F> BackendEigh<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendEigh<F>,
{
    fn eigh_upper(&self, matrix: &Array2<F>) -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>> {
        NdarrayLinAlgBackend.eigh_upper(matrix)
    }
}

// --- Implement BackendQR for Provider ---
#[cfg(feature = "backend_faer")]
impl<F> BackendQR<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendQR<F>,
    faer_specific_code::FaerLinAlgBackend: BackendQR<F>,
{
    fn qr_q_factor(&self, matrix: &Array2<F>) -> Result<Array2<F>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.qr_q_factor(matrix)
    }
}

#[cfg(not(feature = "backend_faer"))]
impl<F> BackendQR<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendQR<F>,
{
    fn qr_q_factor(&self, matrix: &Array2<F>) -> Result<Array2<F>, Box<dyn Error + Send + Sync>> {
        NdarrayLinAlgBackend.qr_q_factor(matrix)
    }
}

// --- Implement BackendSVD for Provider ---
#[cfg(feature = "backend_faer")]
impl<F> BackendSVD<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendSVD<F>,
    faer_specific_code::FaerLinAlgBackend: BackendSVD<F>,
{
    fn svd_into(&self, matrix: Array2<F>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.svd_into(matrix, compute_u, compute_v)
    }
}

#[cfg(not(feature = "backend_faer"))]
impl<F> BackendSVD<F> for LinAlgBackendProvider<F>
where
    F: 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendSVD<F>,
{
    fn svd_into(&self, matrix: Array2<F>, compute_u: bool, compute_v: bool) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>> {
        NdarrayLinAlgBackend.svd_into(matrix, compute_u, compute_v)
    }
}
