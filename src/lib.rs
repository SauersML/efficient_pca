#![cfg_attr(feature = "eigensnp", feature(portable_simd))]
#![doc = include_str!("../README.md")] // Crate-level documentation

pub mod diagnostics;
pub mod linalg_backends; // Consolidated module
pub mod pca;

#[cfg(feature = "eigensnp")]
pub mod eigensnp;

#[cfg(feature = "enable-eigensnp-diagnostics")]
pub mod eigensnp_tests;

pub use pca::PCA;

// Re-export key items from the eigensnp module for users of the EigenSNP functionality.
#[cfg(feature = "eigensnp")]
pub use eigensnp::{
    CondensedFeatureId, EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, EigenSNPCoreOutput,
    LdBlockListId, LdBlockSpecification, PcaReadyGenotypeAccessor, PcaSnpId, PrincipalComponentId,
    QcSampleId,
};
