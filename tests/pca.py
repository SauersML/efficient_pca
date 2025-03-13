"""
This script performs PCA on input data using either scikit-learn (library) or
a custom manual implementation. It provides command-line and optional config-file
control to avoid hard-coded inputs.

Functionality:
1) Accepts input data via CSV file or generates random data if not provided.
2) Allows specifying the number of components (n_components) and other parameters
   via command line or config file.
3) Performs PCA using both:
   - A manual covariance-trick-based implementation
   - The scikit-learn PCA library
4) (Optional) Compares results between the manual and library implementations
   if the --test-flag is set, checking for sign-flip invariances and similarity
   in eigenvalues.
5) Outputs results (transformed data, components, and eigenvalues) in CSV format.
6) Prints copious step-by-step details for clarity and debugging.

Example usages:

  # Basic usage with CSV data file and 2 components:
  python pca_script.py --data_csv mydata.csv --n_components 2

  # Generate random data of shape (10 samples x 5 features), keep 3 components:
  python pca_script.py --samples 10 --features 5 --n_components 3

  # Use a config file (JSON) with fields data_csv, n_components, etc.:
  python pca_script.py --config myconfig.json

  # Compare manual vs library implementation on loaded data:
  python pca_script.py --data_csv data.csv --test-flag

Config file format (JSON), e.g.:
{
  "data_csv": "data.csv",
  "n_components": 2,
  "test_flag": true,
  "samples": 10,
  "features": 5,
  "random_seed": 2025
}

Note:
- Command-line arguments override config-file settings.
- If no data source is specified, random data is generated with default shape
  (5 samples x 5 features).
"""

import argparse
import json
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.linalg as la


def parse_config_file(config_path):
    """
    Parse JSON config file if provided.
    Returns a dictionary of parameters (some may be None if not specified).
    """
    print(f"Attempting to load config file: {config_path}")
    if not os.path.isfile(config_path):
        print(f"Config file {config_path} not found; ignoring.")
        return {}
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config: {config}")
    return config


def parse_arguments():
    """
    Parse command-line arguments.
    Returns a dictionary of parameters.
    """
    parser = argparse.ArgumentParser(description="PCA Script (Manual + Library) with CSV I/O.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to a JSON config file with parameters.")
    parser.add_argument("--data_csv",
                        type=str,
                        help="Path to input CSV containing data matrix (samples x features).")
    parser.add_argument("--n_components",
                        type=int,
                        help="Number of components to keep in PCA.")
    parser.add_argument("--samples",
                        type=int,
                        default=None,
                        help="Number of samples (for random data generation if no CSV).")
    parser.add_argument("--features",
                        type=int,
                        default=None,
                        help="Number of features (for random data generation if no CSV).")
    parser.add_argument("--random_seed",
                        type=int,
                        default=None,
                        help="Random seed for reproducibility in data generation.")
    parser.add_argument("--test-flag",
                        action="store_true",
                        help="Flag to compare manual PCA and library PCA for consistency.")
    args = parser.parse_args()

    # Convert to dict
    cli_params = {
        "config": args.config,
        "data_csv": args.data_csv,
        "n_components": args.n_components,
        "samples": args.samples,
        "features": args.features,
        "random_seed": args.random_seed,
        "test_flag": args.test_flag,
    }
    return cli_params


def load_data_from_csv(csv_path):
    """
    Load data from a CSV file into a NumPy array.
    Assumes rows=samples, columns=features.
    """
    print(f"Loading data from CSV file: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",")
    print(f"Data shape from CSV: {data.shape}")
    return data


def generate_random_data(samples=5, features=5, random_seed=None):
    """
    Generate random data (samples x features).
    """
    print(f"Generating random data with shape ({samples} x {features}).")
    if random_seed is not None:
        print(f"Using random seed: {random_seed}")
        np.random.seed(random_seed)
    data = np.random.randn(samples, features)
    print(f"Random data generated with shape {data.shape}.")
    return data


def manual_pca(X, n_components=None):
    """
    Perform PCA with proper normalization using the covariance trick
    when n_features > n_samples (a 'thin' data matrix).
    
    Returns:
        X_transformed: Transformed data matrix of shape (n_samples, n_components).
        components: Principal component vectors of shape (n_features, n_components).
        eigvals: Eigenvalues associated with each principal component.
    """
    print("[Manual PCA] Starting manual PCA computation...")
    print(f"[Manual PCA] Original shape: {X.shape}")
    
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape
    
    # Determine number of components if not specified
    if n_components is None:
        n_components = min(n_samples, n_features)
    else:
        n_components = min(n_components, min(n_samples, n_features))
    print(f"[Manual PCA] Using n_components={n_components}")
    
    # Apply covariance trick when n_features > n_samples
    if n_features > n_samples:
        print("[Manual PCA] Using the covariance trick (features > samples).")
        gram_matrix = np.dot(X_scaled, X_scaled.T) / (n_samples - 1)
        eigvals, eigvecs = la.eigh(gram_matrix)
        
        # Sort eigenvalues/vectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Select the top components
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
        
        # Calculate the actual principal components
        components = np.zeros((n_features, n_components))
        for i in range(n_components):
            val = eigvals[i]
            scale_factor = np.sqrt(val) if val > 1e-12 else 1e-12
            # V = X^T · U / (sqrt(λ)*(sqrt(n_samples-1)))
            components[:, i] = np.dot(X_scaled.T, eigvecs[:, i]) / (scale_factor * np.sqrt(n_samples - 1))
            # Normalize
            norm_val = np.linalg.norm(components[:, i])
            if norm_val > 1e-12:
                components[:, i] /= norm_val
        
        # Transform data
        X_transformed = np.dot(X_scaled, components)
        
    else:
        print("[Manual PCA] Using the standard covariance matrix approach (samples >= features).")
        cov_matrix = np.dot(X_scaled.T, X_scaled) / (n_samples - 1)
        eigvals, eigvecs = la.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Limit components
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]
        
        # Transform data
        X_transformed = np.dot(X_scaled, eigvecs)
        components = eigvecs
    
    print("[Manual PCA] Finished manual PCA computation.")
    print(f"[Manual PCA] Transformed shape: {X_transformed.shape}")
    return X_transformed, components, eigvals


def library_pca(X, n_components=None):
    """
    Perform PCA using the scikit-learn library.
    
    Returns:
        X_transformed: Transformed data matrix (n_samples x n_components).
        components: Principal components (n_features x n_components).
        explained_variance: Eigenvalues representing variance explained (1D array).
    """
    print("[Library PCA] Starting PCA using scikit-learn...")
    print(f"[Library PCA] Original shape: {X.shape}")
    
    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine max components
    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)
    print(f"[Library PCA] Using n_components={n_components}")
    
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    # scikit-learn returns components_ as shape (n_components, n_features).
    # We transpose it to match the manual_pca's shape (n_features, n_components).
    components = pca.components_.T
    explained_variance = pca.explained_variance_
    
    print("[Library PCA] Finished PCA using scikit-learn.")
    print(f"[Library PCA] Transformed shape: {X_transformed.shape}")
    return X_transformed, components, explained_variance


def compare_pca(X, n_components=None):
    """
    Compare results of manual_pca and library_pca, checking:
      - Transformed data similarity (accounting for sign flips)
      - Eigenvalue similarity (within some tolerance)
    
    Prints comparison details and returns True if they are sufficiently similar,
    False otherwise.
    """
    print("[Comparison] Comparing Manual PCA and Library PCA.")
    manual_transformed, manual_components, manual_eigvals = manual_pca(X, n_components)
    library_transformed, library_components, library_eigvals = library_pca(X, n_components)
    
    # Compare transformed data, allowing sign flips column by column
    print("[Comparison] Checking column-by-column sign-flip invariances in transformed data.")
    if manual_transformed.shape != library_transformed.shape:
        print("[Comparison] Mismatch in shape of transformed data.")
        transformed_similar = False
    else:
        transformed_similar = True
        for i in range(manual_transformed.shape[1]):
            manual_col = manual_transformed[:, i]
            library_col = library_transformed[:, i]
            # Check if close with same sign or opposite sign
            sim_positive = np.allclose(manual_col, library_col, rtol=1e-5, atol=1e-5)
            sim_negative = np.allclose(manual_col, -library_col, rtol=1e-5, atol=1e-5)
            if not (sim_positive or sim_negative):
                print(f"[Comparison] Column {i} differs beyond tolerance.")
                transformed_similar = False
                break
    
    print(f"[Comparison] Transformed data similar (within sign flips): {transformed_similar}")
    
    # Compare eigenvalues
    print("[Comparison] Checking eigenvalues similarity.")
    if len(manual_eigvals) != len(library_eigvals):
        print("[Comparison] Mismatch in number of eigenvalues.")
        eigvals_similar = False
    else:
        eigvals_similar = np.allclose(manual_eigvals, library_eigvals, rtol=1e-5, atol=1e-5)
    print(f"[Comparison] Eigenvalues similar: {eigvals_similar}")
    
    overall = transformed_similar and eigvals_similar
    print(f"[Comparison] Overall PCA match status: {overall}")
    return overall


def output_array_csv(arr, header=""):
    """
    Print a NumPy array as CSV to stdout.
    """
    if header:
        print(header)
    for row in arr:
        if np.ndim(row) == 0:
            # Scalar
            print(f"{row}")
        else:
            print(",".join(str(x) for x in row))


def main():
    print("=== PCA SCRIPT START ===")
    # Set printing options for clarity
    np.set_printoptions(precision=6, suppress=True)

    # Parse command-line arguments
    args = parse_arguments()
    
    # Load config file if provided
    config = {}
    if args["config"]:
        config = parse_config_file(args["config"])

    # Merge config values (config file < command line)
    # Command-line overrides config
    data_csv = args["data_csv"] if args["data_csv"] is not None else config.get("data_csv", None)
    n_components = args["n_components"] if args["n_components"] is not None else config.get("n_components", None)
    test_flag = args["test_flag"] or config.get("test_flag", False)
    samples = args["samples"] if args["samples"] is not None else config.get("samples", None)
    features = args["features"] if args["features"] is not None else config.get("features", None)
    random_seed = args["random_seed"] if args["random_seed"] is not None else config.get("random_seed", None)

    # Decide how to get data
    X = None
    if data_csv:
        print("[Main] Loading data from CSV.")
        X = load_data_from_csv(data_csv)
    else:
        # Generate random data if no CSV specified
        print("[Main] Generating random data (no CSV path specified).")
        if samples is None:
            samples = 5
        if features is None:
            features = 5
        X = generate_random_data(samples, features, random_seed)

    # Now decide whether to run a test (compare manual vs library) or just run library PCA
    if test_flag:
        print("[Main] --test-flag is set. Comparing manual PCA vs library PCA.")
        compare_pca(X, n_components)
    else:
        print("[Main] --test-flag not set. Performing library PCA by default.")
        # Perform library PCA only
        transformed, components, eigvals = library_pca(X, n_components)

        # Output in CSV format
        print("\n--- PCA Output (Library) ---")
        print("\nTransformed Data (CSV):")
        output_array_csv(transformed)

        print("\nComponents (CSV):")
        output_array_csv(components.T)  # Original shape was (n_features, n_components)

        print("\nEigenvalues (CSV):")
        output_array_csv(eigvals.reshape(-1, 1))

    print("=== PCA SCRIPT END ===")


if __name__ == "__main__":
    main()
