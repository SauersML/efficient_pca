// src/linalg_backends.rs

#[derive(Debug, Default, Copy, Clone)]
pub struct LinAlgBackendProvider<F: 'static + Copy + Send + Sync> {
    _phantom: PhantomData<F>,
}

impl<F: 'static + Copy + Send + Sync> LinAlgBackendProvider<F> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

// --- Common imports needed by multiple sections ---
#[cfg(any(
    feature = "backend_openblas",
    feature = "backend_openblas_system",
    feature = "backend_mkl",
    feature = "backend_mkl_system",
    feature = "faer_links_ndarray_static_openblas"
))]
use ndarray::s;
use ndarray::{Array1, Array2};
#[cfg(all(
    not(feature = "backend_faer"),
    any(
        feature = "backend_openblas",
        feature = "backend_openblas_system",
        feature = "backend_mkl",
        feature = "backend_mkl_system",
        feature = "faer_links_ndarray_static_openblas"
    )
))]
use ndarray_linalg::Lapack;
// use num_traits::Float; // No longer needed directly by provider
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
    fn eigh_upper(&self, matrix: &Array2<F>)
        -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>>;
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
    fn svd_into(
        &self,
        matrix: Array2<F>,
        compute_u: bool,
        compute_v: bool,
    ) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>>;
}

// --- NdarrayLinAlgBackend Implementation (originally from ndarray_backend.rs) ---
#[cfg(any(
    feature = "backend_openblas",
    feature = "backend_openblas_system",
    feature = "backend_mkl",
    feature = "backend_mkl_system",
    feature = "faer_links_ndarray_static_openblas"
))]
mod ndarray_backend_impl {
    use super::{s, Array2, BackendEigh, BackendQR, BackendSVD, EighOutput, SVDOutput};
    use ndarray_linalg::{Eigh, Lapack, SVDInto, QR, UPLO};
    use std::error::Error;

    #[cfg_attr(feature = "backend_faer", allow(dead_code))]
    #[derive(Debug, Default, Copy, Clone)]
    pub struct NdarrayLinAlgBackend;

    #[cfg_attr(feature = "backend_faer", allow(dead_code))]
    fn to_dyn_error<E: Error + Send + Sync + 'static>(e: E) -> Box<dyn Error + Send + Sync> {
        Box::new(e)
    }

    impl<F> BackendEigh<F> for NdarrayLinAlgBackend
    where
        F: Lapack<Real = F> + 'static + Copy + Send + Sync,
    {
        fn eigh_upper(
            &self,
            matrix: &Array2<F>,
        ) -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>> {
            let (eigvals, eigvecs) = matrix.eigh(UPLO::Upper).map_err(to_dyn_error)?;
            Ok(EighOutput {
                eigenvalues: eigvals,
                eigenvectors: eigvecs,
            })
        }
    }

    impl<F> BackendQR<F> for NdarrayLinAlgBackend
    where
        F: Lapack + 'static + Copy + Send + Sync,
    {
        fn qr_q_factor(
            &self,
            matrix: &Array2<F>,
        ) -> Result<Array2<F>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if nrows == 0 {
                return Ok(Array2::zeros((0, 0)));
            }
            let k = nrows.min(ncols);
            let (q_full, _) = matrix.qr().map_err(to_dyn_error)?;
            Ok(q_full.slice_move(s![.., 0..k]))
        }
    }

    impl<F> BackendSVD<F> for NdarrayLinAlgBackend
    where
        F: Lapack<Real = F> + 'static + Copy + Send + Sync,
    {
        fn svd_into(
            &self,
            matrix: Array2<F>,
            compute_u: bool,
            compute_v: bool,
        ) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>> {
            let original_rows = matrix.nrows();
            let original_cols = matrix.ncols();

            let (u_option, s, vt_option) = matrix
                .svd_into(compute_u, compute_v)
                .map_err(to_dyn_error)?;

            let k_effective = s.len();

            let u_final = if let Some(mut u_mat) = u_option {
                if u_mat.ncols() > k_effective {
                    assert_eq!(u_mat.nrows(), original_rows, "U matrix row count mismatch");
                    u_mat = u_mat.slice_move(s![.., 0..k_effective]);
                }
                Some(u_mat)
            } else {
                None
            };

            let vt_final = if let Some(mut vt_mat) = vt_option {
                if vt_mat.nrows() > k_effective {
                    assert_eq!(
                        vt_mat.ncols(),
                        original_cols,
                        "VT matrix column count mismatch",
                    );
                    vt_mat = vt_mat.slice_move(s![0..k_effective, ..]);
                }
                Some(vt_mat)
            } else {
                None
            };

            Ok(SVDOutput {
                u: u_final,
                s,
                vt: vt_final,
            })
        }
    }

    pub use NdarrayLinAlgBackend as Backend;
}

#[cfg(all(
    not(feature = "backend_faer"),
    any(
        feature = "backend_openblas",
        feature = "backend_openblas_system",
        feature = "backend_mkl",
        feature = "backend_mkl_system",
        feature = "faer_links_ndarray_static_openblas"
    )
))]
use ndarray_backend_impl::Backend as NdarrayLinAlgBackend;

// --- FaerLinAlgBackend Implementation (originally from faer_backend.rs) ---
#[cfg(feature = "backend_faer")]
mod faer_specific_code {
    // Encapsulate faer-specific code and its imports
    use super::{BackendEigh, BackendQR, BackendSVD, EighOutput, SVDOutput};
    use bytemuck::Pod;
    use faer::traits::num_traits::Zero; // Use Zero via faer's re-export
    use faer::traits::ComplexField;
    use faer::MatRef; // Use faer::MatRef for Faer matrix views.
    use ndarray::{Array1, Array2};
    use std::error::Error;

    // Updated imports for SVD
    // use faer::Parallelism; // No longer needed
    // use faer::dyn_stack::GlobalPodBuffer; // No longer needed
    // use faer::linalg::svd::ComputeSvdVectors as ComputeVectors; // Commented out as likely not needed
    use faer::linalg::solvers::Svd as FaerSolverSvd; // Alias for the new SVD solver
                                                     // SvdReq is likely not needed.

    // --- internal util ---------------------------------------------------------
    #[inline(always)]
    unsafe fn read_unchecked<T: Copy>(ptr: *const T) -> T {
        debug_assert!(!ptr.is_null());
        *ptr
    }

    fn to_dyn_error_faer(msg: String) -> Box<dyn Error + Send + Sync> {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, msg))
    }

    #[derive(Debug, Default, Copy, Clone)]
    pub struct FaerLinAlgBackend;

    fn faer_mat_to_ndarray<F: ComplexField + Copy + Pod + Zero>(
        faer_mat: MatRef<'_, F>,
    ) -> Array2<F> {
        let nrows = faer_mat.nrows();
        let ncols = faer_mat.ncols();
        if nrows == 0 || ncols == 0 {
            return Array2::zeros(ndarray::ShapeBuilder::f((nrows, ncols)));
        }
        let mut data_vec = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                let ptr = unsafe { faer_mat.get_unchecked(i, j) };
                data_vec.push(unsafe { read_unchecked(ptr) });
            }
        }
        Array2::from_shape_vec(ndarray::ShapeBuilder::f((nrows, ncols)), data_vec)
            .expect("Shape and data length mismatch creating ndarray from faer Mat")
    }

    fn faer_col_to_ndarray_vec<F: ComplexField + Copy + Pod + Zero>(
        faer_col: faer::ColRef<'_, F>,
    ) -> Array1<F> {
        let nrows = faer_col.nrows();
        if nrows == 0 {
            return Array1::<F>::zeros(0);
        }
        let mut data_vec = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let ptr = unsafe { faer_col.get_unchecked(i) };
            data_vec.push(unsafe { read_unchecked(ptr) });
        }
        Array1::from_vec(data_vec)
    }

    impl BackendEigh<f64> for FaerLinAlgBackend {
        fn eigh_upper(
            &self,
            matrix: &Array2<f64>,
        ) -> Result<EighOutput<f64>, Box<dyn Error + Send + Sync>> {
            if matrix.nrows() != matrix.ncols() {
                return Err(to_dyn_error_faer(
                    "Matrix must be square for eigendecomposition.".to_string(),
                ));
            }
            if matrix.is_empty() {
                return Ok(EighOutput {
                    eigenvalues: Array1::zeros(0),
                    eigenvectors: Array2::zeros((0, 0)),
                });
            }
            let matrix_view = matrix.view(); // matrix_view is ArrayView2<'_, f64>
            let (nrows, ncols) = matrix_view.dim();

            // --- build a MatRef<'_, f64> that stays alive for the rest of the function ----
            let (faer_mat_view, mut _maybe_owned): (faer::MatRef<'_, f64>, Option<faer::Mat<f64>>); // `_maybe_owned` keeps the buffer alive
            if let Some(slice) = matrix_view.as_slice_memory_order() {
                // Path for contiguous ndarray slice
                faer_mat_view = if matrix_view.is_standard_layout() {
                    // Standard layout means row-major for ndarray
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // Otherwise, it's column-major
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                };
                _maybe_owned = None::<faer::Mat<f64>>; // nothing to free
            } else {
                // not contiguous → make a *column-major* copy
                let col_major_ndarray = matrix_view.to_owned().reversed_axes(); // now F-order
                let temp_owned_faer_mat = faer::MatRef::from_column_major_slice(
                    col_major_ndarray.as_slice().ok_or_else(|| {
                        to_dyn_error_faer(
                            "Failed to get slice from owned column-major copy for Faer Mat".into(),
                        )
                    })?,
                    nrows,
                    ncols,
                )
                .to_owned();
                _maybe_owned = Some(temp_owned_faer_mat);
                faer_mat_view = _maybe_owned
                    .as_ref()
                    .expect("Matrix was just created and put into _maybe_owned")
                    .as_ref();
            }
            let eig = faer_mat_view
                .as_ref()
                .self_adjoint_eigen(faer::Side::Upper)
                .map_err(|e| {
                    to_dyn_error_faer(format!("Faer self_adjoint_eigen failed: {:?}", e))
                })?;
            let eigenvalues_faer_colref = eig.S().column_vector();
            let eigenvectors_faer_matref = eig.U();
            Ok(EighOutput {
                eigenvalues: faer_col_to_ndarray_vec(eigenvalues_faer_colref),
                eigenvectors: faer_mat_to_ndarray(eigenvectors_faer_matref.as_ref()),
            })
        }
    }

    impl BackendQR<f64> for FaerLinAlgBackend {
        fn qr_q_factor(
            &self,
            matrix: &Array2<f64>,
        ) -> Result<Array2<f64>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if nrows == 0 {
                return Ok(Array2::zeros((0, 0)));
            }
            let matrix_view = matrix.view(); // matrix_view is ArrayView2<'_, f64>

            // --- build a MatRef<'_, f64> that stays alive for the rest of the function ----
            let (faer_mat_view, mut _maybe_owned): (faer::MatRef<'_, f64>, Option<faer::Mat<f64>>); // `_maybe_owned` keeps the buffer alive
            if let Some(slice) = matrix_view.as_slice_memory_order() {
                // Path for contiguous ndarray slice
                faer_mat_view = if matrix_view.is_standard_layout() {
                    // Standard layout means row-major for ndarray
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // Otherwise, it's column-major
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                };
                _maybe_owned = None::<faer::Mat<f64>>; // nothing to free
            } else {
                // not contiguous → make a *column-major* copy
                let col_major_ndarray = matrix_view.to_owned().reversed_axes(); // now F-order
                let temp_owned_faer_mat = faer::MatRef::from_column_major_slice(
                    col_major_ndarray.as_slice().ok_or_else(|| {
                        to_dyn_error_faer(
                            "Failed to get slice from owned column-major copy for Faer Mat".into(),
                        )
                    })?,
                    nrows,
                    ncols,
                )
                .to_owned();
                _maybe_owned = Some(temp_owned_faer_mat);
                faer_mat_view = _maybe_owned
                    .as_ref()
                    .expect("Matrix was just created and put into _maybe_owned")
                    .as_ref();
            }
            let qr_decomp = faer_mat_view.as_ref().qr(); // This is faer::MatRef::qr() which returns faer::linalg::solvers::Qr
            let q_thin_faer_mat = qr_decomp.compute_thin_Q(); // This returns an owned Mat<T>
            Ok(faer_mat_to_ndarray(q_thin_faer_mat.as_ref())) // Pass as MatRef
        }
    }

    impl BackendSVD<f64> for FaerLinAlgBackend {
        fn svd_into(
            &self,
            matrix: Array2<f64>,
            compute_u: bool,
            compute_v: bool,
        ) -> Result<SVDOutput<f64>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if matrix.is_empty() {
                let k_dim = nrows.min(ncols);
                return Ok(SVDOutput {
                    u: if compute_u {
                        Some(Array2::zeros((nrows, k_dim)))
                    } else {
                        None
                    },
                    s: Array1::zeros(k_dim),
                    vt: if compute_v {
                        Some(Array2::zeros((k_dim, ncols)))
                    } else {
                        None
                    },
                });
            }
            let matrix_view = matrix.view(); // matrix_view is ArrayView2<'_, f64>

            // --- build a MatRef<'_, f64> that stays alive for the rest of the function ----
            let (faer_mat_view, mut _maybe_owned): (faer::MatRef<'_, f64>, Option<faer::Mat<f64>>); // `_maybe_owned` keeps the buffer alive
            if let Some(slice) = matrix_view.as_slice_memory_order() {
                // Path for contiguous ndarray slice
                faer_mat_view = if matrix_view.is_standard_layout() {
                    // Standard layout means row-major for ndarray
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // Otherwise, it's column-major
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                };
                _maybe_owned = None::<faer::Mat<f64>>; // nothing to free
            } else {
                // not contiguous → make a *column-major* copy
                let col_major_ndarray = matrix_view.to_owned().reversed_axes(); // now F-order
                let temp_owned_faer_mat = faer::MatRef::from_column_major_slice(
                    col_major_ndarray.as_slice().ok_or_else(|| {
                        to_dyn_error_faer(
                            "Failed to get slice from owned column-major copy for Faer Mat".into(),
                        )
                    })?,
                    nrows,
                    ncols,
                )
                .to_owned();
                _maybe_owned = Some(temp_owned_faer_mat);
                faer_mat_view = _maybe_owned
                    .as_ref()
                    .expect("Matrix was just created and put into _maybe_owned")
                    .as_ref();
            }
            let faer_mat_ref = faer_mat_view.as_ref();

            let svd_solver_instance = FaerSolverSvd::new_thin(faer_mat_ref)
                .map_err(|e| to_dyn_error_faer(format!("Faer SVD computation failed: {:?}", e)))?;

            let s_diag_ref = svd_solver_instance.S();
            let s_ndarray = faer_col_to_ndarray_vec(s_diag_ref.column_vector());

            let u_ndarray = if compute_u {
                Some(faer_mat_to_ndarray(svd_solver_instance.U().as_ref()))
            } else {
                None
            };

            let vt_ndarray = if compute_v {
                let v_mat_ref = svd_solver_instance.V();
                let v_ndarray = faer_mat_to_ndarray(v_mat_ref.as_ref());
                Some(v_ndarray.t().into_owned())
            } else {
                None
            };

            Ok(SVDOutput {
                u: u_ndarray,
                s: s_ndarray,
                vt: vt_ndarray,
            })
        }
    }

    impl BackendEigh<f32> for FaerLinAlgBackend {
        fn eigh_upper(
            &self,
            matrix: &Array2<f32>,
        ) -> Result<EighOutput<f32>, Box<dyn Error + Send + Sync>> {
            if matrix.nrows() != matrix.ncols() {
                return Err(to_dyn_error_faer(
                    "Matrix must be square for eigendecomposition.".to_string(),
                ));
            }
            if matrix.is_empty() {
                return Ok(EighOutput {
                    eigenvalues: Array1::zeros(0),
                    eigenvectors: Array2::zeros((0, 0)),
                });
            }
            let matrix_view = matrix.view(); // matrix_view is ArrayView2<'_, f32>
            let (nrows, ncols) = matrix_view.dim();

            // --- build a MatRef<'_, f32> that stays alive for the rest of the function ----
            let (faer_mat_view, mut _maybe_owned): (faer::MatRef<'_, f32>, Option<faer::Mat<f32>>); // `_maybe_owned` keeps the buffer alive
            if let Some(slice) = matrix_view.as_slice_memory_order() {
                // Path for contiguous ndarray slice
                faer_mat_view = if matrix_view.is_standard_layout() {
                    // Standard layout means row-major for ndarray
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // Otherwise, it's column-major
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                };
                _maybe_owned = None::<faer::Mat<f32>>; // nothing to free
            } else {
                // not contiguous → make a *column-major* copy
                let col_major_ndarray = matrix_view.to_owned().reversed_axes(); // now F-order
                let temp_owned_faer_mat = faer::MatRef::from_column_major_slice(
                    col_major_ndarray.as_slice().ok_or_else(|| {
                        to_dyn_error_faer(
                            "Failed to get slice from owned column-major copy for Faer Mat".into(),
                        )
                    })?,
                    nrows,
                    ncols,
                )
                .to_owned();
                _maybe_owned = Some(temp_owned_faer_mat);
                faer_mat_view = _maybe_owned
                    .as_ref()
                    .expect("Matrix was just created and put into _maybe_owned")
                    .as_ref();
            }
            let eig = faer_mat_view
                .as_ref()
                .self_adjoint_eigen(faer::Side::Upper)
                .map_err(|e| {
                    to_dyn_error_faer(format!("Faer self_adjoint_eigen failed: {:?}", e))
                })?;
            let eigenvalues_faer_colref = eig.S().column_vector();
            let eigenvectors_faer_matref = eig.U();
            Ok(EighOutput {
                eigenvalues: faer_col_to_ndarray_vec(eigenvalues_faer_colref),
                eigenvectors: faer_mat_to_ndarray(eigenvectors_faer_matref.as_ref()),
            })
        }
    }

    impl BackendQR<f32> for FaerLinAlgBackend {
        fn qr_q_factor(
            &self,
            matrix: &Array2<f32>,
        ) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if nrows == 0 {
                return Ok(Array2::zeros((0, 0)));
            }
            let matrix_view = matrix.view(); // matrix_view is ArrayView2<'_, f32>

            // --- build a MatRef<'_, f32> that stays alive for the rest of the function ----
            let (faer_mat_view, mut _maybe_owned): (faer::MatRef<'_, f32>, Option<faer::Mat<f32>>); // `_maybe_owned` keeps the buffer alive
            if let Some(slice) = matrix_view.as_slice_memory_order() {
                // Path for contiguous ndarray slice
                faer_mat_view = if matrix_view.is_standard_layout() {
                    // Standard layout means row-major for ndarray
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // Otherwise, it's column-major
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                };
                _maybe_owned = None::<faer::Mat<f32>>; // nothing to free
            } else {
                // not contiguous → make a *column-major* copy
                let col_major_ndarray = matrix_view.to_owned().reversed_axes(); // now F-order
                let temp_owned_faer_mat = faer::MatRef::from_column_major_slice(
                    col_major_ndarray.as_slice().ok_or_else(|| {
                        to_dyn_error_faer(
                            "Failed to get slice from owned column-major copy for Faer Mat".into(),
                        )
                    })?,
                    nrows,
                    ncols,
                )
                .to_owned();
                _maybe_owned = Some(temp_owned_faer_mat);
                faer_mat_view = _maybe_owned
                    .as_ref()
                    .expect("Matrix was just created and put into _maybe_owned")
                    .as_ref();
            }
            let qr_decomp = faer_mat_view.as_ref().qr(); // This is faer::MatRef::qr() which returns faer::linalg::solvers::Qr
            let q_thin_faer_mat = qr_decomp.compute_thin_Q(); // This returns an owned Mat<T>
            Ok(faer_mat_to_ndarray(q_thin_faer_mat.as_ref())) // Pass as MatRef
        }
    }

    impl BackendSVD<f32> for FaerLinAlgBackend {
        fn svd_into(
            &self,
            matrix: Array2<f32>,
            compute_u: bool,
            compute_v: bool,
        ) -> Result<SVDOutput<f32>, Box<dyn Error + Send + Sync>> {
            let (nrows, ncols) = matrix.dim();
            if matrix.is_empty() {
                let k_dim = nrows.min(ncols);
                return Ok(SVDOutput {
                    u: if compute_u {
                        Some(Array2::zeros((nrows, k_dim)))
                    } else {
                        None
                    },
                    s: Array1::zeros(k_dim),
                    vt: if compute_v {
                        Some(Array2::zeros((k_dim, ncols)))
                    } else {
                        None
                    },
                });
            }
            let matrix_view = matrix.view(); // matrix_view is ArrayView2<'_, f32>

            // --- build a MatRef<'_, f32> that stays alive for the rest of the function ----
            let (faer_mat_view, mut _maybe_owned): (faer::MatRef<'_, f32>, Option<faer::Mat<f32>>); // `_maybe_owned` keeps the buffer alive
            if let Some(slice) = matrix_view.as_slice_memory_order() {
                // Path for contiguous ndarray slice
                faer_mat_view = if matrix_view.is_standard_layout() {
                    // Standard layout means row-major for ndarray
                    faer::MatRef::from_row_major_slice(slice, nrows, ncols)
                } else {
                    // Otherwise, it's column-major
                    faer::MatRef::from_column_major_slice(slice, nrows, ncols)
                };
                _maybe_owned = None::<faer::Mat<f32>>; // nothing to free
            } else {
                // not contiguous → make a *column-major* copy
                let col_major_ndarray = matrix_view.to_owned().reversed_axes(); // now F-order
                let temp_owned_faer_mat = faer::MatRef::from_column_major_slice(
                    col_major_ndarray.as_slice().ok_or_else(|| {
                        to_dyn_error_faer(
                            "Failed to get slice from owned column-major copy for Faer Mat".into(),
                        )
                    })?,
                    nrows,
                    ncols,
                )
                .to_owned();
                _maybe_owned = Some(temp_owned_faer_mat);
                faer_mat_view = _maybe_owned
                    .as_ref()
                    .expect("Matrix was just created and put into _maybe_owned")
                    .as_ref();
            }
            let faer_mat_ref = faer_mat_view.as_ref();

            let svd_solver_instance = FaerSolverSvd::new_thin(faer_mat_ref)
                .map_err(|e| to_dyn_error_faer(format!("Faer SVD computation failed: {:?}", e)))?;

            let s_diag_ref = svd_solver_instance.S();
            let s_ndarray = faer_col_to_ndarray_vec(s_diag_ref.column_vector());

            let u_ndarray = if compute_u {
                Some(faer_mat_to_ndarray(svd_solver_instance.U().as_ref()))
            } else {
                None
            };

            let vt_ndarray = if compute_v {
                let v_mat_ref = svd_solver_instance.V();
                let v_ndarray = faer_mat_to_ndarray(v_mat_ref.as_ref());
                Some(v_ndarray.t().into_owned())
            } else {
                None
            };

            Ok(SVDOutput {
                u: u_ndarray,
                s: s_ndarray,
                vt: vt_ndarray,
            })
        }
    }
} // End of faer_specific_code module

// --- LinAlgBackendProvider Dispatch (originally from linalg_backend_dispatch.rs) ---

// Import concrete backend types for the provider
#[cfg(feature = "backend_faer")]
// NdarrayLinAlgBackend is already defined in this file.

/// A provider struct that dispatches to the selected linear algebra backend
/// based on compile-time feature flags.

// --- Implement BackendEigh for Provider ---
#[cfg(feature = "backend_faer")]
impl BackendEigh<f32> for LinAlgBackendProvider<f32>
where
    faer_specific_code::FaerLinAlgBackend: BackendEigh<f32>,
{
    fn eigh_upper(
        &self,
        matrix: &Array2<f32>,
    ) -> Result<EighOutput<f32>, Box<dyn Error + Send + Sync>> {
        // Assuming FaerLinAlgBackend is default constructible or has a new()
        faer_specific_code::FaerLinAlgBackend.eigh_upper(matrix)
    }
}

#[cfg(feature = "backend_faer")]
impl BackendEigh<f64> for LinAlgBackendProvider<f64>
where
    faer_specific_code::FaerLinAlgBackend: BackendEigh<f64>,
{
    fn eigh_upper(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<EighOutput<f64>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.eigh_upper(matrix)
    }
}

#[cfg(not(feature = "backend_faer"))]
impl<F> BackendEigh<F> for LinAlgBackendProvider<F>
where
    F: Lapack<Real = F> + 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendEigh<F>,
{
    fn eigh_upper(
        &self,
        matrix: &Array2<F>,
    ) -> Result<EighOutput<F>, Box<dyn Error + Send + Sync>> {
        NdarrayLinAlgBackend.eigh_upper(matrix)
    }
}

// --- Implement BackendQR for Provider ---
#[cfg(feature = "backend_faer")]
impl BackendQR<f32> for LinAlgBackendProvider<f32>
// The where clause `faer_specific_code::FaerLinAlgBackend: BackendQR<f32>`
// can be omitted as the call itself will enforce it, but keeping it is also fine.
// For consistency with Eigh/SVD, let's keep it for now.
where
    faer_specific_code::FaerLinAlgBackend: BackendQR<f32>,
{
    fn qr_q_factor(
        &self,
        matrix: &Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.qr_q_factor(matrix)
    }
}

#[cfg(feature = "backend_faer")]
impl BackendQR<f64> for LinAlgBackendProvider<f64>
where
    faer_specific_code::FaerLinAlgBackend: BackendQR<f64>,
{
    fn qr_q_factor(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.qr_q_factor(matrix)
    }
}

#[cfg(not(feature = "backend_faer"))]
impl<F> BackendQR<F> for LinAlgBackendProvider<F>
where
    F: Lapack + 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendQR<F>,
{
    fn qr_q_factor(&self, matrix: &Array2<F>) -> Result<Array2<F>, Box<dyn Error + Send + Sync>> {
        NdarrayLinAlgBackend.qr_q_factor(matrix)
    }
}

// --- Implement BackendSVD for Provider ---
#[cfg(feature = "backend_faer")]
impl BackendSVD<f32> for LinAlgBackendProvider<f32>
where
    faer_specific_code::FaerLinAlgBackend: BackendSVD<f32>,
{
    fn svd_into(
        &self,
        matrix: Array2<f32>,
        compute_u: bool,
        compute_v: bool,
    ) -> Result<SVDOutput<f32>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.svd_into(matrix, compute_u, compute_v)
    }
}

#[cfg(feature = "backend_faer")]
impl BackendSVD<f64> for LinAlgBackendProvider<f64>
where
    faer_specific_code::FaerLinAlgBackend: BackendSVD<f64>,
{
    fn svd_into(
        &self,
        matrix: Array2<f64>,
        compute_u: bool,
        compute_v: bool,
    ) -> Result<SVDOutput<f64>, Box<dyn Error + Send + Sync>> {
        faer_specific_code::FaerLinAlgBackend.svd_into(matrix, compute_u, compute_v)
    }
}

#[cfg(not(feature = "backend_faer"))]
impl<F> BackendSVD<F> for LinAlgBackendProvider<F>
where
    F: Lapack<Real = F> + 'static + Copy + Send + Sync,
    NdarrayLinAlgBackend: BackendSVD<F>,
{
    fn svd_into(
        &self,
        matrix: Array2<F>,
        compute_u: bool,
        compute_v: bool,
    ) -> Result<SVDOutput<F>, Box<dyn Error + Send + Sync>> {
        NdarrayLinAlgBackend.svd_into(matrix, compute_u, compute_v)
    }
}
