use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use efficient_pca::PCA;
use ndarray::{Array, Array2}; // Ensure Array is imported for Array::random
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// Function to generate random data for PCA
fn generate_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    Array::random((n_samples, n_features), Uniform::new(0., 10.))
}

// Benchmark for PCA::fit
fn bench_pca_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA_fit");

    for &(n_samples, n_features) in [(100, 50), (500, 100), (100, 200)].iter() {
        let data = generate_data(n_samples, n_features);
        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::new("fit", format!("{}x{}", n_samples, n_features)),
            &(data.clone(), None), 
            |b, (data_matrix, tolerance)| {
                b.iter_with_setup(
                    || (PCA::new(), data_matrix.clone()), 
                    |(mut pca, data_to_fit)| pca.fit(data_to_fit, *tolerance).unwrap(),
                );
            },
        );
    }
    group.finish();
}

// Benchmark for PCA::rfit
fn bench_pca_rfit(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA_rfit");

    let n_components_requested_base = 10;
    let n_oversamples = 5;
    let seed = Some(42u64);
    let tolerance_rfit = Some(1e-5);

    for &(n_samples, n_features) in [(500, 200), (1000, 500), (200, 1000)].iter() {
        let data = generate_data(n_samples, n_features);
        let n_components_actual = n_components_requested_base.min(n_samples.min(n_features));
        if n_components_actual == 0 { 
            continue;
        }

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));
        group.bench_with_input(
            BenchmarkId::new("rfit", format!("{}x{}", n_samples, n_features)),
            &data.clone(),
            |b, data_matrix| {
                b.iter_with_setup(
                    || (PCA::new(), data_matrix.clone()), 
                    |(mut pca, data_to_fit)| {
                        pca.rfit(
                            data_to_fit,
                            n_components_actual,
                            n_oversamples,
                            seed,
                            tolerance_rfit,
                        )
                        .unwrap();
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_pca_fit, bench_pca_rfit);
criterion_main!(benches);
