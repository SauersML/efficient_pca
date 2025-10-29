# EigenSNP: Large-Scale Genomic PCA

EigenSNP is a sophisticated Principal Component Analysis algorithm specifically designed for large-scale genomic datasets, such as those found in biobanks or population-scale studies. It efficiently handles datasets where the number of SNPs (features) significantly exceeds the number of samples.

To use EigenSNP, enable the `eigensnp` feature in your `Cargo.toml` (note that this currently requires the use of nightly Rust):
```toml
[dependencies]
efficient_pca = { version = "*", features = ["eigensnp"] }
```

## Key Features

* 🧬 **Genomic-Optimized**: Designed for SNP data with linkage disequilibrium (LD) block structure
* ⚡ **Scalable**: Uses randomized SVD (RSVD) and memory-efficient f32 precision for large datasets
* 🔧 **Highly Configurable**: Extensive tuning parameters for different dataset characteristics
* 📊 **Multi-Stage Algorithm**: Local eigenSNP basis learning → condensed features → global PCA → refinement
* 🔍 **Diagnostics**: Optional detailed diagnostics for algorithm analysis (with feature flag)
* 💾 **Output Options**: Can save local PC loadings per LD block to TSV files

## Algorithm Overview

EigenSNP processes genomic data through several stages:

1. **Local Basis Learning**: For each LD block, learns local eigenSNP basis vectors using RSVD on a subset of samples
2. **Condensed Feature Construction**: Projects all samples onto local bases to create a condensed feature matrix
3. **Feature Standardization**: Standardizes the condensed features (zero mean, unit variance)
4. **Global PCA**: Applies RSVD to the standardized condensed features to get initial PC scores
5. **Refinement**: Iteratively refines SNP loadings and sample scores through orthogonalization and SVD

## Core Types

### `EigenSNPCoreAlgorithm`
The main orchestrator that executes the EigenSNP workflow.

```rust
use efficient_pca::eigensnp::{EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig};

let config = EigenSNPCoreAlgorithmConfig {
    target_num_global_pcs: 10,
    components_per_ld_block: 5,
    ..Default::default()
};
let algorithm = EigenSNPCoreAlgorithm::new(config);
```

### `EigenSNPCoreAlgorithmConfig`
Configuration parameters for fine-tuning the algorithm:

```rust
use efficient_pca::eigensnp::EigenSNPCoreAlgorithmConfig;

let config = EigenSNPCoreAlgorithmConfig {
    // Global PCA parameters
    target_num_global_pcs: 15,                    // Number of final PCs to compute
    global_pca_sketch_oversampling: 10,           // Extra dimensions for RSVD sketching
    global_pca_num_power_iterations: 2,           // Power iterations for global RSVD

    // Local basis learning parameters
    subset_factor_for_local_basis_learning: 0.1,  // Fraction of samples for local learning
    min_subset_size_for_local_basis_learning: 20_000,
    max_subset_size_for_local_basis_learning: 60_000,
    components_per_ld_block: 7,                   // Local PCs per LD block

    // RSVD parameters for local learning
    local_rsvd_sketch_oversampling: 4,
    local_rsvd_num_power_iterations: 2,

    // Processing parameters
    snp_processing_strip_size: 2000,              // SNPs per processing chunk
    refine_pass_count: 1,                         // Number of refinement iterations
    random_seed: 2025,

    // Optional outputs
    collect_diagnostics: false,                   // Requires "enable-eigensnp-diagnostics" feature
    local_pcs_output_dir: None,                   // Directory to save local PC loadings

    ..Default::default()
};
```

### `PcaReadyGenotypeAccessor`
Trait for providing access to standardized genotype data:

```rust
use efficient_pca::eigensnp::{PcaReadyGenotypeAccessor, PcaSnpId, QcSampleId, ThreadSafeStdError};
use ndarray::Array2;

struct MyGenotypeData {
    standardized_data: Array2<f32>, // SNPs x Samples, already standardized
}

impl PcaReadyGenotypeAccessor for MyGenotypeData {
    fn get_standardized_snp_sample_block(
        &self,
        snp_ids: &[PcaSnpId],
        sample_ids: &[QcSampleId],
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        // Return requested SNP x Sample block from your standardized data
        // Implementation depends on your data storage format
        unimplemented!()
    }

    fn num_pca_snps(&self) -> usize {
        self.standardized_data.nrows()
    }

    fn num_qc_samples(&self) -> usize {
        self.standardized_data.ncols()
    }
}
```

### `LdBlockSpecification`
Defines linkage disequilibrium blocks:

```rust
use efficient_pca::eigensnp::{LdBlockSpecification, PcaSnpId};

let ld_blocks = vec![
    LdBlockSpecification {
        user_defined_block_tag: "chr1_block1".to_string(),
        pca_snp_ids_in_block: (0..1000).map(PcaSnpId).collect(),
    },
    LdBlockSpecification {
        user_defined_block_tag: "chr1_block2".to_string(),
        pca_snp_ids_in_block: (1000..2000).map(PcaSnpId).collect(),
    },
    // ... more blocks
];
```

### `EigenSNPCoreOutput`
Final results containing PCA components:

```rust,ignore
// After running compute_pca
let (output, diagnostics) = algorithm.compute_pca(&genotype_data, &ld_blocks, &snp_metadata)?;

// Access results
let loadings = &output.final_snp_principal_component_loadings;  // SNPs x PCs
let scores = &output.final_sample_principal_component_scores;   // Samples x PCs
let eigenvalues = &output.final_principal_component_eigenvalues; // PC eigenvalues
let num_pcs = output.num_principal_components_computed;
```

## Usage Example

```rust
use efficient_pca::eigensnp::{
    EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig,
    LdBlockSpecification, PcaReadyGenotypeAccessor, PcaSnpId,
    PcaSnpMetadata, QcSampleId, ThreadSafeStdError,
};
use ndarray::{array, Array2};
use std::sync::Arc;

struct InMemoryAccessor {
    data: Array2<f32>,
}

impl PcaReadyGenotypeAccessor for InMemoryAccessor {
    fn get_standardized_snp_sample_block(
        &self,
        snp_ids: &[PcaSnpId],
        sample_ids: &[QcSampleId],
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let mut block = Array2::zeros((snp_ids.len(), sample_ids.len()));
        for (row_idx, PcaSnpId(snp_idx)) in snp_ids.iter().enumerate() {
            for (col_idx, QcSampleId(sample_idx)) in sample_ids.iter().enumerate() {
                block[(row_idx, col_idx)] = self.data[(*snp_idx, *sample_idx)];
            }
        }
        Ok(block)
    }

    fn num_pca_snps(&self) -> usize {
        self.data.nrows()
    }

    fn num_qc_samples(&self) -> usize {
        self.data.ncols()
    }
}

# fn main() -> Result<(), ThreadSafeStdError> {
let config = EigenSNPCoreAlgorithmConfig {
    target_num_global_pcs: 2,
    components_per_ld_block: 2,
    subset_factor_for_local_basis_learning: 1.0,
    min_subset_size_for_local_basis_learning: 1,
    max_subset_size_for_local_basis_learning: 10,
    ..Default::default()
};

let algorithm = EigenSNPCoreAlgorithm::new(config);

let ld_blocks = vec![LdBlockSpecification {
    user_defined_block_tag: "block1".to_string(),
    pca_snp_ids_in_block: vec![PcaSnpId(0), PcaSnpId(1)],
}];

let snp_metadata = vec![
    PcaSnpMetadata {
        id: Arc::new("snp1".to_string()),
        chr: Arc::new("1".to_string()),
        pos: 100,
    },
    PcaSnpMetadata {
        id: Arc::new("snp2".to_string()),
        chr: Arc::new("1".to_string()),
        pos: 200,
    },
];

let genotype_data = InMemoryAccessor {
    data: array![
        [0.5, -0.5, 1.0],
        [-0.5, 0.5, -1.0],
    ],
};

let (output, _diagnostics) = algorithm.compute_pca(
    &genotype_data,
    &ld_blocks,
    &snp_metadata,
)?;

println!(
    "Computed {} principal components",
    output.num_principal_components_computed
);
println!(
    "SNP loadings shape: {:?}",
    output.final_snp_principal_component_loadings.dim()
);
println!(
    "Sample scores shape: {:?}",
    output.final_sample_principal_component_scores.dim()
);
# Ok(())
# }
```

## Performance Considerations

* **Memory Efficiency**: Uses `f32` precision with `f64` accumulation for critical operations
* **Large Datasets**: Designed for datasets with millions of SNPs and thousands to hundreds of thousands of samples
* **LD Block Size**: Optimal LD block sizes depend on your data; typically 100-5000 SNPs per block
* **Subset Size**: For local basis learning, uses 10-50% of samples by default (configurable)
* **RSVD Parameters**: Increase oversampling and power iterations for higher accuracy at computational cost

## Diagnostics and Debugging

Enable detailed diagnostics with the `enable-eigensnp-diagnostics` feature:

```toml
[dependencies]
efficient_pca = { version = "*", features = ["enable-eigensnp-diagnostics"] }
```

```rust
use efficient_pca::eigensnp::EigenSNPCoreAlgorithmConfig;

let _config = EigenSNPCoreAlgorithmConfig {
    collect_diagnostics: true,
    local_pcs_output_dir: Some("./local_pcs_output".to_string()),
    ..Default::default()
};
```

This provides detailed algorithm metrics and saves local PC loadings to TSV files for inspection.

## Index Types

EigenSNP uses strongly-typed indices for safety:

* **`PcaSnpId(usize)`**: Identifies SNPs in the final PCA-ready dataset
* **`QcSampleId(usize)`**: Identifies quality-controlled samples
* **`LdBlockListId(usize)`**: Identifies LD blocks in the specification list
* **`CondensedFeatureId(usize)`**: Identifies rows in the condensed feature matrix
* **`PrincipalComponentId(usize)`**: Identifies computed principal components
