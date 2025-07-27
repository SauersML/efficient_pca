# Efficient Principal Component Analysis in Rust

This Rust library provides Principal Component Analysis (PCA), both exact and fast approximate methods. It is a modified version of the original work by Erik Garrison. Forked from https://github.com/ekg/pca.

---
## Core Features

* ✨ **Exact PCA (`fit`)**: Computes principal components via eigen-decomposition of the covariance matrix. For datasets where the number of features is greater than the number of samples (`n_features > n_samples`), it uses the Gram matrix method (the "Gram trick"). Allows for component selection based on an eigenvalue tolerance.
* ⚡ **Randomized PCA (`rfit`)**: Employs a memory-efficient randomized SVD algorithm to approximate principal components.
* 🛡️ **Data Handling**:
    * Input data is automatically mean-centered.
    * Feature scaling is applied using standard deviations. Scale factors are sanitized to always be positive.
    * Computed principal component vectors are normalized to unit length.
* 💾 **Model Persistence**: Fitted PCA models (including mean, scale factors, and rotation matrix) can be saved to and loaded from files using `bincode` serialization.
* 🔄 **Data Transformation**: Once a PCA model is fitted or loaded, it can be used to transform new data into the principal component space. This transformation also applies the learned centering and scaling.

---
## Installation

Add `efficient_pca` to your `Cargo.toml` dependencies.

```sh
cargo add efficient_pca
```

### Linear Algebra Backend Selection

This crate supports multiple linear algebra backends for optimal performance across different platforms:

#### OpenBLAS Backends
- **`backend_openblas`** (default): Uses statically linked OpenBLAS
- **`backend_openblas_system`**: Uses system/dynamically linked OpenBLAS
  - **Recommended for macOS** where static linking can be problematic
  - Requires OpenBLAS to be installed on your system

#### Intel MKL Backends  
- **`backend_mkl`**: Uses statically linked Intel MKL
- **`backend_mkl_system`**: Uses system/dynamically linked Intel MKL

#### Alternative Backend
- **`backend_faer`**: Uses the pure Rust `faer` linear algebra library

#### Usage Examples

For macOS users or when experiencing static linking issues:
```toml
[dependencies]
efficient_pca = { version = "*", default-features = false, features = ["backend_openblas_system"] }
```

For Intel systems with MKL installed:
```toml
[dependencies]
efficient_pca = { version = "*", default-features = false, features = ["backend_mkl_system"] }
```

For pure Rust environments:
```toml
[dependencies]
efficient_pca = { version = "*", default-features = false, features = ["backend_faer"] }
```

---
## API Overview

### `PCA::new()`
Creates a new, empty `PCA` struct. The model is not fitted and needs to be computed using `fit` or `rfit`, or loaded.

### `PCA::with_model(rotation, mean, raw_standard_deviations)`
Creates a `PCA` instance from pre-computed components (rotation matrix, mean vector, and raw standard deviations).
* `raw_standard_deviations`: Input standard deviations for each feature. Values `s` that are non-finite or where `s <= 1e-9` are sanitized to `1.0` before being stored. This makes sure the internal scale factors are always positive and finite. An error is returned if input `raw_standard_deviations` initially contains non-finite values.

### `PCA::fit(&mut self, data_matrix, tolerance)`
Computes the PCA model using an exact method.
* `data_matrix`: The input data (`n_samples` x `n_features`).
* `tolerance`: Optional `f64`. If `Some(tol_val)`, principal components corresponding to eigenvalues less than `tol_val * max_eigenvalue` are discarded. `tol_val` is clamped to `[0.0, 1.0]`. If `None`, all components up to the matrix rank are retained.

### `PCA::rfit(&mut self, x_input_data, n_components_requested, n_oversamples, seed, tol)`
Computes an approximate PCA model using a memory-efficient randomized SVD algorithm and returns the transformed principal component scores for `x_input_data`.
* `x_input_data`: The input data (`n_samples` x `n_features`). This matrix is modified in place for centering and scaling.
* `n_components_requested`: The target number of principal components to compute and keep.
* `n_oversamples`: Number of additional random dimensions (`p`) to sample for the sketch (`l = k + p`).
    * If `0`, an adaptive default for `p` is used (typically 10% of `n_components_requested`, clamped between 5 and 20).
    * If positive, this value is used, but an internal minimum is enforced for robustness. Recommended explicit values: 5-20.
* `seed`: Optional `u64` for the random number generator.
* `tol`: Optional `f64` (typically between 0.0 and 1.0, exclusive). If `Some(t_val)` where `0.0 < t_val < 1.0`, components are further filtered if their corresponding singular value `s_i` from the internal SVD of the projected sketch satisfies `s_i <= t_val * s_max`.

### `PCA::transform(&self, x)`
Applies the learned PCA transformation (centering, scaling, and rotation) to new data `x`.
* `x`: Input data to transform (`m_samples` x `d_features`). This matrix is modified in place during centering and scaling.

### `PCA::rotation(&self)`
Returns an `Option<&Array2<f64>>` to the rotation matrix (principal components), if computed. Shape: (`n_features`, `k_components`).

### `PCA::explained_variance(&self)`
Returns an `Option<&Array1<f64>>` to the explained variance for each principal component, if computed.

### `PCA::save_model(&self, path)`
Saves the current PCA model (rotation, mean, scale, and optionally explained_variance) to the specified file path using `bincode` serialization.

### `PCA::load_model(path)`
Loads a PCA model from a file previously saved with `save_model`. The loaded model is validated for completeness and internal consistency (e.g., matching dimensions, positive scale factors).

---
## Performance Considerations

* **`fit()`**: Provides exact PCA. It's generally suitable for datasets where the smaller dimension (either samples or features) is not excessively large, allowing for direct eigen-decomposition. It automatically uses the Gram matrix optimization if `n_features > n_samples`.
* **`rfit()`**: A significant speed-up and reduced memory footprint for very large or high-dimensional datasets where an approximation of PCA is acceptable. The accuracy is typically good.

---
## Authors and Acknowledgements

* This library is a fork and modification of the original `pca` crate by Erik Garrison (original repository: <https://github.com/ekg/pca>).
* Extended by SauersML.

---
## EigenSNP: Large-Scale Genomic PCA

EigenSNP is a sophisticated Principal Component Analysis algorithm specifically designed for large-scale genomic datasets, such as those found in biobanks or population-scale studies. It efficiently handles datasets where the number of SNPs (features) significantly exceeds the number of samples.

### Key Features

* 🧬 **Genomic-Optimized**: Designed for SNP data with linkage disequilibrium (LD) block structure
* ⚡ **Scalable**: Uses randomized SVD (RSVD) and memory-efficient f32 precision for large datasets  
* 🔧 **Highly Configurable**: Extensive tuning parameters for different dataset characteristics
* 📊 **Multi-Stage Algorithm**: Local eigenSNP basis learning → condensed features → global PCA → refinement
* 🔍 **Diagnostics**: Optional detailed diagnostics for algorithm analysis (with feature flag)
* 💾 **Output Options**: Can save local PC loadings per LD block to TSV files

### Algorithm Overview

EigenSNP processes genomic data through several stages:

1. **Local Basis Learning**: For each LD block, learns local eigenSNP basis vectors using RSVD on a subset of samples
2. **Condensed Feature Construction**: Projects all samples onto local bases to create a condensed feature matrix
3. **Feature Standardization**: Standardizes the condensed features (zero mean, unit variance)
4. **Global PCA**: Applies RSVD to the standardized condensed features to get initial PC scores
5. **Refinement**: Iteratively refines SNP loadings and sample scores through orthogonalization and SVD

### Core Types

#### `EigenSNPCoreAlgorithm`
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

#### `EigenSNPCoreAlgorithmConfig`
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

#### `PcaReadyGenotypeAccessor`
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

#### `LdBlockSpecification`
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

#### `EigenSNPCoreOutput`
Final results containing PCA components:

```rust
// After running compute_pca
let (output, diagnostics) = algorithm.compute_pca(&genotype_data, &ld_blocks, &snp_metadata)?;

// Access results
let loadings = &output.final_snp_principal_component_loadings;  // SNPs x PCs
let scores = &output.final_sample_principal_component_scores;   // Samples x PCs  
let eigenvalues = &output.final_principal_component_eigenvalues; // PC eigenvalues
let num_pcs = output.num_principal_components_computed;
```

### Usage Example

```rust
use efficient_pca::eigensnp::{
    EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, 
    LdBlockSpecification, PcaSnpMetadata, PcaSnpId
};
use std::sync::Arc;

// Configure the algorithm
let config = EigenSNPCoreAlgorithmConfig {
    target_num_global_pcs: 10,
    components_per_ld_block: 5,
    subset_factor_for_local_basis_learning: 0.15,
    ..Default::default()
};

let algorithm = EigenSNPCoreAlgorithm::new(config);

// Define your LD blocks (example: simple non-overlapping blocks)
let ld_blocks = create_ld_blocks_from_your_data();

// Provide SNP metadata
let snp_metadata: Vec<PcaSnpMetadata> = your_snps.iter().map(|snp| {
    PcaSnpMetadata {
        id: Arc::new(snp.id.clone()),
        chr: Arc::new(snp.chromosome.clone()),
        pos: snp.position,
    }
}).collect();

// Your genotype data accessor (implements PcaReadyGenotypeAccessor)
let genotype_data = MyGenotypeAccessor::new(your_standardized_genotype_matrix);

// Run EigenSNP PCA
let (output, _diagnostics) = algorithm.compute_pca(
    &genotype_data, 
    &ld_blocks,
    &snp_metadata
)?;

// Use the results
println!("Computed {} principal components", output.num_principal_components_computed);
println!("SNP loadings shape: {:?}", output.final_snp_principal_component_loadings.dim());
println!("Sample scores shape: {:?}", output.final_sample_principal_component_scores.dim());
```

### Performance Considerations

* **Memory Efficiency**: Uses `f32` precision with `f64` accumulation for critical operations
* **Large Datasets**: Designed for datasets with millions of SNPs and thousands to hundreds of thousands of samples
* **LD Block Size**: Optimal LD block sizes depend on your data; typically 100-5000 SNPs per block
* **Subset Size**: For local basis learning, uses 10-50% of samples by default (configurable)
* **RSVD Parameters**: Increase oversampling and power iterations for higher accuracy at computational cost

### Diagnostics and Debugging

Enable detailed diagnostics with the `enable-eigensnp-diagnostics` feature:

```toml
[dependencies]
efficient_pca = { version = "*", features = ["enable-eigensnp-diagnostics"] }
```

```rust
let config = EigenSNPCoreAlgorithmConfig {
    collect_diagnostics: true,
    local_pcs_output_dir: Some("./local_pcs_output".to_string()),
    ..Default::default()
};
```

This provides detailed algorithm metrics and saves local PC loadings to TSV files for inspection.

### Index Types

EigenSNP uses strongly-typed indices for safety:

* **`PcaSnpId(usize)`**: Identifies SNPs in the final PCA-ready dataset
* **`QcSampleId(usize)`**: Identifies quality-controlled samples  
* **`LdBlockListId(usize)`**: Identifies LD blocks in the specification list
* **`CondensedFeatureId(usize)`**: Identifies rows in the condensed feature matrix
* **`PrincipalComponentId(usize)`**: Identifies computed principal components

---
## License

This project is licensed under the MIT License.
