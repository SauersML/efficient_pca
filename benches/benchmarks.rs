// Global allocator setup for jemalloc
#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use efficient_pca::PCA;
use ndarray::Array2;
use rand::distributions::{Uniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::time::Instant;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use jemalloc_ctl::{epoch, stats};

// Enum to specify the type of data source for benchmarks.
#[derive(Clone, Debug)]
enum DataSource {
    Dense012,
    Sparse012(f64), // Parameter is sparsity level (e.g., 0.95 for 95% zeros)
    LowVariance012 {
        fraction_low_var_feats: f64, // Proportion of features that are low-variance
        majority_val_in_low_var_feat: f64, // In a low-var feature, probability of seeing the 'majority value'
                                           // The 'majority value' itself will be fixed (e.g. 0.0) for simplicity in this generator.
                                           // Minority values will be 1.0 or 2.0, split equally.
    },
}


/// Generates random data of shape (n_samples x n_features) with values 0, 1, or 2 (as f64), seeded for reproducibility.
fn generate_random_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let distribution = Uniform::new_inclusive(0, 2);
    Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.sample(distribution) as f64
    })
}

/// Generates sparse random data of shape (n_samples x n_features).
/// Values are 0.0 (with probability `sparsity`), or 1.0/2.0 (with equal probability for the remainder).
fn generate_sparse_random_data(
    n_samples: usize,
    n_features: usize,
    sparsity: f64,
    seed: u64,
) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let value_distribution = Uniform::new_inclusive(1, 2); // For 1.0 or 2.0

    Array2::from_shape_fn((n_samples, n_features), |_| {
        if rng.gen_range(0.0..1.0) < sparsity {
            0.0
        } else {
            rng.sample(value_distribution) as f64
        }
    })
}

/// Generates low-variance random data.
/// For a `fraction_low_var_feats` of columns, values are `0.0` with probability `majority_val_in_low_var_feat_freq`,
/// or `1.0`/`2.0` (equal chance) otherwise. Other columns are standard 0,1,2 random.
fn generate_low_variance_data(
    n_samples: usize,
    n_features: usize,
    fraction_low_var_feats: f64,
    majority_val_in_low_var_feat_freq: f64,
    seed: u64,
) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let standard_dist = Uniform::new_inclusive(0, 2);
    let minority_val_dist = Uniform::new_inclusive(1, 2);

    let mut data_vec = Vec::with_capacity(n_samples * n_features);

    for _j in 0..n_features {
        let is_low_var_feature = rng.gen::<f64>() < fraction_low_var_feats;
        for _i in 0..n_samples {
            if is_low_var_feature {
                if rng.gen::<f64>() < majority_val_in_low_var_feat_freq {
                    data_vec.push(0.0);
                } else {
                    data_vec.push(rng.sample(minority_val_dist) as f64);
                }
            } else {
                data_vec.push(rng.sample(standard_dist) as f64);
            }
        }
    }
    Array2::from_shape_vec((n_samples, n_features), data_vec)
        .expect("Shape mismatch in generate_low_variance_data")
}


/// Formats a memory size in KB into KB/MB/GB with 3 decimals where reasonable.
fn format_memory_kb(mem_kb: u64) -> String {
    if mem_kb < 1024 {
        return format!("{} KB", mem_kb);
    }
    let mb = mem_kb as f64 / 1024.0;
    if mb < 1024.0 {
        return format!("{:.3} MB", mb);
    }
    let gb = mb / 1024.0;
    format!("{:.3} GB", gb)
}

/// Runs `fit` or `rfit` on the provided data, returns (time_secs, memory_kb) measured.
fn benchmark_pca(
    use_rfit: bool,
    data: &Array2<f64>,
    n_components_override: Option<usize>,
    n_oversamples_for_rfit: usize,
    seed_for_rfit: u64,
) -> (f64, u64, u64) {
    // Memory stats using jemalloc_ctl
    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    epoch::advance().unwrap();
    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    let resident_before = stats::resident::read().unwrap();
    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    let active_before = stats::active::read().unwrap(); // active is closer to virtual memory used by application

    // Fallback for non-jemalloc or msvc builds - RSS and Virt will be 0
    #[cfg(not(all(feature = "jemalloc", not(target_env = "msvc"))))]
    let resident_before = 0;
    #[cfg(not(all(feature = "jemalloc", not(target_env = "msvc"))))]
    let active_before = 0;

    let start_time = Instant::now();

    let mut pca = PCA::new();
    let transformed_data: Array2<f64>;

    let n_components_to_use_for_rfit = n_components_override.unwrap_or_else(|| {
        std::cmp::min(data.nrows(), data.ncols()).min(30).max(2)
    });

    if use_rfit {
        transformed_data = pca
            .rfit(
                data.clone(),
                n_components_to_use_for_rfit,
                n_oversamples_for_rfit,
                Some(seed_for_rfit),
                None,
            )
            .expect("rfit failed");
        assert_eq!(transformed_data.ncols(), n_components_to_use_for_rfit, "RFIT: Transformed data column count should match requested components for rfit.");
    } else {
        pca.fit(data.clone(), None).expect("fit failed");
        transformed_data = pca.transform(data.clone()).expect("transform failed");
        let actual_fit_components = pca.rotation().map_or(0, |r| r.ncols());
        assert_eq!(transformed_data.ncols(), actual_fit_components, "FIT: Transformed data column count should match actual components in the model after fit.");
    }

    assert_eq!(transformed_data.nrows(), data.nrows(), "Transformed data should have same number of rows as input.");

    let duration = start_time.elapsed().as_secs_f64();

    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    epoch::advance().unwrap();
    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    let resident_after = stats::resident::read().unwrap();
    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    let active_after = stats::active::read().unwrap();

    #[cfg(not(all(feature = "jemalloc", not(target_env = "msvc"))))]
    let resident_after = 0;
    #[cfg(not(all(feature = "jemalloc", not(target_env = "msvc"))))]
    let active_after = 0;

    let rss_delta_bytes = resident_after.saturating_sub(resident_before);
    let virt_delta_bytes = active_after.saturating_sub(active_before);

    (duration, (rss_delta_bytes / 1024) as u64, (virt_delta_bytes / 1024) as u64) // Convert bytes to KB
}



fn write_raw_data_to_tsv(
    raw_data: &[RawBenchDataPoint],
    filename: &str,
) -> Result<(), std::io::Error> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "ScenarioName	NumSamples	NumFeatures	BackendName	Iteration	RunType	TimeSec	RSSDeltaKB	VirtDeltaKB	NumComponentsOverride"
    )?;

    // Write data
    for point in raw_data {
        let n_comp_str = point.n_components_override
                            .map_or_else(|| "None".to_string(), |k| k.to_string());
        writeln!(
            writer,
            "{}	{}	{}	{}	{}	{}	{:.6}	{}	{}	{}", // Using {:.6} for TimeSec for precision
            point.scenario_name,
            point.n_samples,
            point.n_features,
            point.backend_name,
            point.iteration_idx,
            point.run_type,
            point.time_sec,
            point.rss_delta_kb,
            point.virt_delta_kb,
            n_comp_str
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct RawBenchDataPoint {
    scenario_name: String,
    n_samples: usize,
    n_features: usize,
    backend_name: String,
    iteration_idx: u64, // Criterion's iteration count (from 0 to iters-1)
    run_type: String,    // "fit" or "rfit"
    time_sec: f64,
    rss_delta_kb: u64,
    virt_delta_kb: u64,
    n_components_override: Option<usize>,
}

fn determine_appropriate_sample_size(
    scenario_name_short: &str, // e.g., "Large", "Wide", "Small"
    is_rfit: bool,
    _n_samples: usize, // _ to indicate potentially unused for now, but good for context
    n_features: usize
) -> usize {
    if !is_rfit { // Fit method
        match scenario_name_short {
            "Large" | "Square" | "Sparse-W" => return 10,
            "Wide" | "LowVar-W" | "Wide-k10" | "Wide-k50" | "Wide-k200" if n_features >= 10000 => return 10,
            "Wide-XL" if n_features >= 100000 => return 10,
            "Wide-L" if n_features >= 50000 => return 20,
            "Medium" | "Tall" => return 50,
            _ => return 100, // For "Small" and other faster scenarios
        }
    } else { // Rfit method
        match scenario_name_short {
            "Wide-k200" if n_features >= 10000 => return 20,
            "Wide-XL" if n_features >= 100000 => return 30,
            "Large" | "Wide-L" | "Sparse-W" => return 50,
            _ => return 100, // For other faster rfit scenarios
        }
    }
}

fn criterion_benchmark_runner(c: &mut Criterion) {
    let mut all_raw_data = Vec::<RawBenchDataPoint>::new();
    let current_backend_name = if cfg!(feature = "backend_faer") { "faer".to_string() } else { "ndarray".to_string() };

    let scenarios = vec![
        ("Small", 100, 50, 1234, DataSource::Dense012, None),
        ("Medium", 1000, 500, 1234, DataSource::Dense012, None),
        ("Large", 5000, 2000, 1234, DataSource::Dense012, None),
        ("Square", 2000, 2000, 1234, DataSource::Dense012, None),
        ("Tall", 10000, 500, 1234, DataSource::Dense012, None),
        ("Wide", 500, 10000, 1234, DataSource::Dense012, None),
        ("Wide-L", 100, 50000, 1234, DataSource::Dense012, None),
        ("Wide-XL", 88, 100000, 1234, DataSource::Dense012, None),
        ("Sparse-W", 500, 20000, 1234, DataSource::Sparse012(0.95), None),
        ("LowVar-W", 500, 10000, 1234, DataSource::LowVariance012 { fraction_low_var_feats: 0.5, majority_val_in_low_var_feat: 0.95 }, None),
        ("Wide-k10", 500, 10000, 1234, DataSource::Dense012, Some(10)),
        ("Wide-k50", 500, 10000, 1234, DataSource::Dense012, Some(50)),
        ("Wide-k200", 500, 10000, 1234, DataSource::Dense012, Some(200)),
    ];


    for (name, n_samples, n_features, seed, data_source_type, n_components_override) in scenarios {
        // Clone data_source_type if it's captured by multiple closures or used after move.
        let data = match data_source_type.clone() {
            DataSource::Dense012 => generate_random_data(n_samples, n_features, seed),
            DataSource::Sparse012(s) => generate_sparse_random_data(n_samples, n_features, s, seed),
            DataSource::LowVariance012 { fraction_low_var_feats, majority_val_in_low_var_feat } =>
                generate_low_variance_data(n_samples, n_features, fraction_low_var_feats, majority_val_in_low_var_feat, seed),
        };

        let oversamples_for_rfit = 0;

        // --- FIT Benchmark ---


        // --- FIT Benchmark ---
        let fit_group_name = format!("fit/{}", name);
        let fit_sample_size = determine_appropriate_sample_size(name, false, n_samples, n_features);
        let mut fit_group = c.benchmark_group(fit_group_name);
        fit_group.sample_size(fit_sample_size);
        let input_size_bytes = (n_samples * n_features * std::mem::size_of::<f64>()) as u64;
        fit_group.throughput(Throughput::Bytes(input_size_bytes));

        let _fit_benchmark_id = BenchmarkId::new(
            "fit", // Use "fit" as function_id
            format!("{}_s{}_f{}_c{:?}", name, n_samples, n_features, n_components_override) // Parameter string
        );

        fit_group.bench_with_input(_fit_benchmark_id, &data.clone(), |b, data_to_bench| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);
                for i in 0..iters { // Use 'i' for iteration_idx
                    let (time_taken, rss_mem_used, virt_mem_used) = benchmark_pca(false, data_to_bench, n_components_override, oversamples_for_rfit, seed);
                    total_duration += std::time::Duration::from_secs_f64(time_taken);
                    
                    all_raw_data.push(RawBenchDataPoint {
                        scenario_name: name.to_string(),
                        n_samples,
                        n_features,
                        backend_name: current_backend_name.clone(),
                        iteration_idx: i,
                        run_type: "fit".to_string(),
                        time_sec: time_taken,
                        rss_delta_kb: rss_mem_used,
                        virt_delta_kb: virt_mem_used,
                        n_components_override,
                    });

                }
                total_duration
            });
        });
        fit_group.finish();
        

        // --- RFIT Benchmark ---
        let rfit_group_name = format!("rfit/{}", name);
        let rfit_sample_size = determine_appropriate_sample_size(name, true, n_samples, n_features);
        let mut rfit_group = c.benchmark_group(rfit_group_name);
        rfit_group.sample_size(rfit_sample_size);
        rfit_group.throughput(Throughput::Bytes(input_size_bytes));

        let rfit_benchmark_id = BenchmarkId::new(
            "rfit", // Use "rfit" as function_id
            format!("{}_s{}_f{}_c{:?}", name, n_samples, n_features, n_components_override) // Parameter string
        );
        
        rfit_group.bench_with_input(rfit_benchmark_id, &data.clone(), |b, data_to_bench| {
             b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0,0);
                for i in 0..iters { // Use 'i' for iteration_idx
                    let (time_taken, rss_mem_used, virt_mem_used) = benchmark_pca(true, data_to_bench, n_components_override, oversamples_for_rfit, seed);
                    total_duration += std::time::Duration::from_secs_f64(time_taken);

                    all_raw_data.push(RawBenchDataPoint {
                        scenario_name: name.to_string(),
                        n_samples,
                        n_features,
                        backend_name: current_backend_name.clone(),
                        iteration_idx: i,
                        run_type: "rfit".to_string(),
                        time_sec: time_taken,
                        rss_delta_kb: rss_mem_used,
                        virt_delta_kb: virt_mem_used,
                        n_components_override,
                    });
                    
                }
                total_duration
            });
        });
        rfit_group.finish();

    }
    

    if let Err(e) = write_raw_data_to_tsv(&all_raw_data, "benchmark_raw_results.tsv") {
        eprintln!("Failed to write raw benchmark data to TSV: {}", e);
    }
}

criterion_group!(benches, criterion_benchmark_runner);
criterion_main!(benches);
