use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use efficient_pca::PCA;
use ndarray::Array2;
use rand::distributions::{Uniform}; // Added Distribution for Uniform
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant; // Keep for benchmark_pca internal timing, though Criterion handles overall.
use sysinfo::System; // Added SystemExt, ProcessExt, PidExt for sysinfo

// Enum to specify the type of data source for benchmarks.
#[derive(Clone, Debug)] // Added Clone and Debug for DataSource
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

// Holds results for a single benchmark scenario.
#[derive(Debug)] // Added Debug for BenchResult
struct BenchResult {
    scenario_name: String,
    n_samples: usize,
    n_features: usize,
    fit_time: f64,
    fit_memory_kb: u64,
    rfit_time: f64,
    rfit_memory_kb: u64,
    backend_name: String,
    n_components_override: Option<usize>,
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
) -> (f64, u64) {
    let mut sys = System::new_all();
    sys.refresh_all();
    let pid = sysinfo::get_current_pid().expect("Unable to get current PID");
    let process_start = sys.process(pid).expect("Unable to get current process");
    let initial_mem = process_start.memory();

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

    sys.refresh_all();
    let process_end = sys
        .process(pid)
        .expect("Unable to get current process at end");
    let final_mem = process_end.memory();
    let used = if final_mem > initial_mem {
        final_mem - initial_mem
    } else {
        0
    };

    (duration, used / 1024) // Convert bytes to KB
}

/// Appends benchmark results to a CSV file. Creates the file and writes headers if it doesn't exist.
fn append_results_to_csv(
    results: &[BenchResult],
    filename: &str,
) -> Result<(), std::io::Error> {
    let path = Path::new(filename);
    let file_exists = path.exists();

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;

    if !file_exists {
        writeln!(
            file,
            "Scenario,Samples,Features,Backend,FitTimeSec,FitMemoryKB,RFitTimeSec,RFitMemoryKB,NumComponentsOverride"
        )?;
    }

    for r in results {
        let n_comp_str = r.n_components_override.map_or_else(|| "Default".to_string(), |k| k.to_string());
        writeln!(
            file,
            "{},{},{},{},{:.3},{},{:.3},{},{}",
            r.scenario_name,
            r.n_samples,
            r.n_features,
            r.backend_name,
            r.fit_time,
            r.fit_memory_kb,
            r.rfit_time,
            r.rfit_memory_kb,
            n_comp_str
        )?;
    }
    Ok(())
}

/// Prints a final summary table of all scenario results.
fn print_summary_table(results: &[BenchResult]) {
    println!("\n===== FINAL SUMMARY TABLE (from CSV data) =====");
    println!(
        "{:<10} | {:>8} | {:>8} | {:<7} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
        "Scenario", "Samples", "Features", "Backend", "fit (s)", "fit Mem", "rfit (s)", "rfit Mem", "CompsReq"
    );
    println!(
        "----------+----------+----------+---------+------------+------------+------------+------------+----------"
    );
    for r in results {
        let n_comp_disp = r.n_components_override.map_or_else(|| "Def".to_string(), |k| k.to_string());
        println!(
            "{:<10} | {:>8} | {:>8} | {:<7} | {:>10.3} | {:>10} | {:>10.3} | {:>10} | {:>8}",
            r.scenario_name,
            r.n_samples,
            r.n_features,
            r.backend_name,
            r.fit_time,
            format_memory_kb(r.fit_memory_kb),
            r.rfit_time,
            format_memory_kb(r.rfit_memory_kb),
            n_comp_disp
        );
    }
    println!("==========================================================================================");
}


fn criterion_benchmark_runner(c: &mut Criterion) {
    let scenarios = vec![
        ("Small", 100, 50, 1234, DataSource::Dense012, None),
        ("Medium", 1000, 500, 1234, DataSource::Dense012, None),
        ("Large", 5000, 2000, 1234, DataSource::Dense012, None),
        ("Square", 2000, 2000, 1234, DataSource::Dense012, None),
        ("Tall", 10000, 500, 1234, DataSource::Dense012, None),
        ("Wide", 500, 10000, 1234, DataSource::Dense012, None),
        ("Wide-L", 100, 50000, 1234, DataSource::Dense012, None),
        ("Wide-XL", 88, 100000, 1234, DataSource::Dense012, None), // Adjusted samples to match test
        ("Sparse-W", 500, 20000, 1234, DataSource::Sparse012(0.95), None),
        ("LowVar-W", 500, 10000, 1234, DataSource::LowVariance012 { fraction_low_var_feats: 0.5, majority_val_in_low_var_feat: 0.95 }, None),
        ("Wide-k10", 500, 10000, 1234, DataSource::Dense012, Some(10)),
        ("Wide-k50", 500, 10000, 1234, DataSource::Dense012, Some(50)),
        ("Wide-k200", 500, 10000, 1234, DataSource::Dense012, Some(200)),
    ];

    let mut collected_results_for_csv = Vec::new();
    let mut group = c.benchmark_group("PCA Benchmarks"); // Renamed group

    for (name, n_samples, n_features, seed, data_source_type, n_components_override) in scenarios {
        // Clone data_source_type if it's captured by multiple closures or used after move.
        let data = match data_source_type.clone() {
            DataSource::Dense012 => generate_random_data(n_samples, n_features, seed),
            DataSource::Sparse012(s) => generate_sparse_random_data(n_samples, n_features, s, seed),
            DataSource::LowVariance012 { fraction_low_var_feats, majority_val_in_low_var_feat } =>
                generate_low_variance_data(n_samples, n_features, fraction_low_var_feats, majority_val_in_low_var_feat, seed),
        };

        let oversamples_for_rfit = 10;

        // --- FIT Benchmark ---
        let fit_benchmark_id = BenchmarkId::new(
            "fit", // Use "fit" as function_id
            format!("{}_s{}_f{}_c{:?}", name, n_samples, n_features, n_components_override) // Parameter string
        );
        let mut fit_time_manual = 0.0;
        let mut fit_mem_kb_manual = 0;

        // Set throughput for FIT
        let input_size_bytes = (n_samples * n_features * std::mem::size_of::<f64>()) as u64;
        group.throughput(Throughput::Bytes(input_size_bytes));

        group.bench_with_input(fit_benchmark_id, &data.clone(), |b, data_to_bench| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0,0);
                for _i in 0..iters {
                    let (time_taken, mem_used) = benchmark_pca(false, data_to_bench, n_components_override, oversamples_for_rfit, seed);
                    total_duration += std::time::Duration::from_secs_f64(time_taken);
                    if fit_time_manual == 0.0 {
                        fit_time_manual = time_taken;
                        fit_mem_kb_manual = mem_used;
                    }
                }
                total_duration
            });
        });

        // --- RFIT Benchmark ---
        let rfit_benchmark_id = BenchmarkId::new(
            "rfit", // Use "rfit" as function_id
            format!("{}_s{}_f{}_c{:?}", name, n_samples, n_features, n_components_override) // Parameter string
        );
        let mut rfit_time_manual = 0.0;
        let mut rfit_mem_kb_manual = 0;
        
        // Set throughput for RFIT (same as FIT)
        group.throughput(Throughput::Bytes(input_size_bytes));

        group.bench_with_input(rfit_benchmark_id, &data.clone(), |b, data_to_bench| {
             b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0,0);
                for _i in 0..iters {
                    let (time_taken, mem_used) = benchmark_pca(true, data_to_bench, n_components_override, oversamples_for_rfit, seed);
                    total_duration += std::time::Duration::from_secs_f64(time_taken);
                    if rfit_time_manual == 0.0 {
                        rfit_time_manual = time_taken;
                        rfit_mem_kb_manual = mem_used;
                    }
                }
                total_duration
            });
        });

        let backend_name = if cfg!(feature = "backend_faer") { "faer".to_string() } else { "ndarray".to_string() };
        collected_results_for_csv.push(BenchResult {
            scenario_name: name.to_string(),
            n_samples,
            n_features,
            fit_time: fit_time_manual,
            fit_memory_kb: fit_mem_kb_manual,
            rfit_time: rfit_time_manual,
            rfit_memory_kb: rfit_mem_kb_manual,
            backend_name,
            n_components_override,
        });
    }
    
    group.finish(); // Finish group

    // After all benchmarks, print and save CSV
    print_summary_table(&collected_results_for_csv);
    if let Err(e) = append_results_to_csv(&collected_results_for_csv, "benchmark_results.csv") {
        eprintln!("Failed to write benchmark results to CSV: {}", e);
    }
}

criterion_group!(benches, criterion_benchmark_runner);
criterion_main!(benches);
