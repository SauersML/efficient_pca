// src/eigensnp_tests.rs

// This module is conditionally compiled via src/lib.rs
// #[cfg(feature = "enable-eigensnp-diagnostics")]

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize}; // For TestResultRecord if it needs to be serialized later
use std::sync::Mutex; // Using once_cell for static initialization

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResultRecord {
    pub test_name: String,
    pub num_features_d: usize,
    pub num_samples_n: usize,
    pub num_components_k: usize,
    pub num_ld_blocks_b: usize,
    pub components_per_block_c: usize,
    pub success: bool,
    pub outcome_details: String,
    pub notes: String,
    pub approx_peak_mem_mibs: Option<f64>,
    pub computation_time_secs: Option<f64>,
}

pub static TEST_RESULTS: Lazy<Mutex<Vec<TestResultRecord>>> = Lazy::new(|| Mutex::new(Vec::new()));

// Helper function to add a result, could be used by tests.
// This is not strictly required by the import error but useful for the pattern.
#[allow(dead_code)]
pub fn add_test_result(record: TestResultRecord) {
    match TEST_RESULTS.lock() {
        Ok(mut guard) => guard.push(record),
        Err(poisoned) => {
            // Handle the case where the Mutex is poisoned
            // For example, log an error or recover the data
            eprintln!("Failed to lock TEST_RESULTS: Mutex was poisoned. Trying to recover.");
            let mut guard = poisoned.into_inner(); // Get the inner data
            guard.push(record); // Try to push data anyway
        }
    }
}
