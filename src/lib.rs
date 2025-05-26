#![feature(where_clause_attrs)]
#![doc = include_str!("../README.md")] // Crate-level documentation

#[cfg(all(feature = "backend_mkl", feature = "backend_openblas"))]
compile_error!("The 'backend_mkl' and 'backend_openblas' features are mutually exclusive. If 'backend_mkl' is enabled, make sure project default features are disabled using '--no-default-features'.");

pub mod linalg_backends; // Consolidated module
pub mod pca;
pub mod eigensnp;


pub use pca::PCA;

// Re-export key items from the eigensnp module for users of the EigenSNP functionality.
pub use eigensnp::{
    EigenSNPCoreAlgorithm,
    EigenSNPCoreAlgorithmConfig,
    EigenSNPCoreOutput,
    LdBlockSpecification,
    PcaReadyGenotypeAccessor,
    PcaSnpId,
    QcSampleId,
    LdBlockListId,
    CondensedFeatureId,
    PrincipalComponentId,
};
