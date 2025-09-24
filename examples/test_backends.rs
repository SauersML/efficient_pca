use efficient_pca::PCA;
use ndarray::Array2;

fn main() {
    // Create a simple test matrix
    let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Create and fit PCA
    let mut pca = PCA::new();
    pca.fit(data, None).expect("PCA fit failed");

    println!("PCA backend test works!");
    println!("Rotation matrix shape: {:?}", pca.rotation().unwrap().dim());
    println!(
        "Explained variance: {:?}",
        pca.explained_variance().unwrap()
    );
}
