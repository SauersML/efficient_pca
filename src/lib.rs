#![doc = include_str!("../README.md")] // Crate-level documentation

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

#[cfg(test)]
mod pca_tests;
mod eigensnp_tests;
