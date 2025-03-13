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
   if the --test-flag is set:
     - If a CSV is supplied, it will test that particular dataset.
     - Itt will also run a battery of random tests.
5) Outputs results (transformed data, components, and eigenvalues) in CSV format.
6) Prints copious step-by-step details for clarity and debugging.

Example usages:

  # Basic usage with CSV data file and 2 components:
  python pca.py --data_csv mydata.csv --n_components 2

  # Generate random data of shape (10 samples x 5 features), keep 3 components:
  python pca.py --samples 10 --features 5 --n_components 3

  # Use a config file (JSON) with fields data_csv, n_components, etc.:
  python pca.py --config myconfig.json

  # Compare manual vs library implementation on loaded data:
  python pca.py --data_csv data.csv --test-flag

  # If --test-flag is provided without a CSV, the script will run multiple random tests:
  python pca.py --test-flag

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
- When --test-flag is used, the script compares manual vs. library PCA and also
  runs a battery of random-dimension tests.
"""

import argparse
import json
import numpy as np
import os
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random


def parse_config_file(config_path):
    """
    Parse JSON config file if provided.
    Returns a dictionary of parameters (some may be None if not specified).
    """
    print(f"[Config] Attempting to load config file: {config_path}")
    if not os.path.isfile(config_path):
        print(f"[Config] Config file '{config_path}' not found; ignoring.")
        return {}
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"[Config] Loaded config: {config}")
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
                        help="Compare manual PCA and library PCA for consistency and run a suite of tests.")
    parser.add_argument("--human-readable",
                        action="store_true",
                        help="Print data and results in a more verbose/human-readable format (otherwise CSV).")

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
        "human_readable": args.human_readable,
    }
    return cli_params


def load_data_from_csv(csv_path):
    """
    Load data from a CSV file into a NumPy array.
    Assumes rows=samples, columns=features.
    """
    print(f"[Data] Loading data from CSV file: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",")
    print(f"[Data] Data shape from CSV: {data.shape}")
    return data


def generate_random_data(samples=5, features=5, random_seed=None):
    """
    Generate random data (samples x features).
    If random_seed is provided, we use it for reproducibility;
    otherwise data is non-deterministic.
    """
    print(f"[Data] Generating random data with shape ({samples} x {features}).")
    if random_seed is not None:
        print(f"[Data] Using random seed: {random_seed}")
        np.random.seed(random_seed)
    data = np.random.randn(samples, features)
    print(f"[Data] Random data generated with shape {data.shape}.")
    return data


def manual_pca(X, n_components=None):
    """
    Perform PCA with proper normalization using the covariance trick
    when n_features > n_samples (a 'thin' data matrix).

    Returns:
        X_transformed: Transformed data matrix (n_samples, n_components).
        components: Principal component vectors (n_features, n_components).
        eigvals: Eigenvalues associated with each principal component.
    """
    print("[Manual PCA] Starting manual PCA computation...")
    print(f"[Manual PCA] Original data shape: {X.shape}")

    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape
    print("[Manual PCA] Data has been standardized (zero mean, unit variance).")

    # Determine the number of components if not specified
    if n_components is None:
        n_components = min(n_samples, n_features)
    else:
        n_components = min(n_components, min(n_samples, n_features))
    print(f"[Manual PCA] Using n_components = {n_components}")

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
            if val < 1e-12:
                # For zero variance, return a zero vector
                components[:, i] = 0.0
            else:
                scale_factor = np.sqrt(val)
                components[:, i] = (X_scaled.T @ eigvecs[:, i]) / (scale_factor * np.sqrt(n_samples - 1))
                # Normalize each component
                comp_norm = np.linalg.norm(components[:, i])
                if comp_norm > 1e-12:
                    components[:, i] /= comp_norm
                else:
                    print(f"[Manual PCA] Warning: Component {i} norm is near zero.")

        # Transform data
        X_transformed = X_scaled @ components

    else:
        print("[Manual PCA] Using standard covariance approach (samples >= features).")
        cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)
        eigvals, eigvecs = la.eigh(cov_matrix)

        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Limit components
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

        # Transform data
        X_transformed = X_scaled @ eigvecs
        components = eigvecs

    print(f"[Manual PCA] PCA finished. Transformed shape: {X_transformed.shape}")
    return X_transformed, components, eigvals


def library_pca(X, n_components=None):
    """
    Perform PCA using the scikit-learn library.

    Returns:
        X_transformed: Transformed data matrix (n_samples, n_components).
        components: Principal components (n_features, n_components).
        explained_variance: The eigenvalues representing variance explained.
    """
    print("[Library PCA] Starting PCA using scikit-learn...")
    print(f"[Library PCA] Original data shape: {X.shape}")

    # Center and scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape
    print("[Library PCA] Data has been standardized (zero mean, unit variance).")

    # Determine max components
    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)
    print(f"[Library PCA] Using n_components = {n_components}")

    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    # scikit-learn returns components_ in shape (n_components, n_features).
    # We'll transpose to match the manual PCA shape of (n_features, n_components).
    components = pca.components_.T
    explained_variance = pca.explained_variance_

    print(f"[Library PCA] PCA finished. Transformed shape: {X_transformed.shape}")
    return X_transformed, components, explained_variance


def compare_pca(X, n_components=None):
    """
    Compare results of manual_pca and library_pca, checking:
      - Transformed data similarity (accounting for sign flips)
      - Eigenvalue similarity (within a tolerance)

    Prints comparison details and returns True if they are sufficiently similar,
    False otherwise.
    """
    print("[Comparison] Comparing Manual PCA and Library PCA.")
    manual_transformed, manual_components, manual_eigvals = manual_pca(X, n_components)
    library_transformed, library_components, library_eigvals = library_pca(X, n_components)
    
    # Calculate and print explained variance
    total_variance_manual = np.sum(manual_eigvals)
    total_variance_library = np.sum(library_eigvals)
    
    print("\n[Comparison] Explained Variance:")
    print("Component | Manual PCA |  %  | Library PCA |  %  ")
    print("---------+------------+-----+-------------+-----")
    for i in range(len(manual_eigvals)):
        manual_pct = (manual_eigvals[i] / total_variance_manual) * 100
        library_pct = (library_eigvals[i] / total_variance_library) * 100
        print(f"    PC{i+1}  | {manual_eigvals[i]:10.2f} | {manual_pct:3.1f}% | {library_eigvals[i]:11.2f} | {library_pct:3.1f}%")
    
    print(f"\nTotal variance - Manual: {total_variance_manual:.2f}, Library: {total_variance_library:.2f}")
    print(f"PC1 captures {(manual_eigvals[0] / total_variance_manual) * 100:.1f}% of variance in Manual PCA")
    print(f"PC1 captures {(library_eigvals[0] / total_variance_library) * 100:.1f}% of variance in Library PCA")

    # Compare shapes
    same_shape = (manual_transformed.shape == library_transformed.shape)
    if not same_shape:
        print("[Comparison] Mismatch in shape of transformed data.")
        return False

    # Compare transformed data, allowing sign flips column by column
    print("[Comparison] Checking column-by-column sign-flip invariance in transformed data...")
    transformed_similar = True
    for i in range(manual_transformed.shape[1]):
        manual_col = manual_transformed[:, i]
        library_col = library_transformed[:, i]

        # Check if close with same sign or opposite sign
        sim_positive = np.allclose(manual_col, library_col, rtol=1e-5, atol=1e-5)
        sim_negative = np.allclose(manual_col, -library_col, rtol=1e-5, atol=1e-5)

        if not (sim_positive or sim_negative):
            print(f"[Comparison] Column {i} differs beyond tolerance.")
            print("[Comparison] Manual implementation (ACTUAL):")
            print(manual_col)
            print("[Comparison] Library implementation (EXPECTED):")
            print(library_col)
            print("[Comparison] Absolute difference:")
            print(np.abs(manual_col - library_col))
            print("[Comparison] Max difference:", np.max(np.abs(manual_col - library_col)))
            transformed_similar = False
            break

    # Compare eigenvalues
    eigvals_similar = False
    if len(manual_eigvals) == len(library_eigvals):
        eigvals_similar = np.allclose(manual_eigvals, library_eigvals, rtol=1e-5, atol=1e-5)

    print("[Comparison] Results:")
    print(f"  -> Transformed data similar (accounting for sign flips): {transformed_similar}")
    print(f"  -> Eigenvalues similar: {eigvals_similar}")

    overall = transformed_similar and eigvals_similar
    print(f"[Comparison] Overall PCA match status: {overall}")
    
    # If comparison failed, print full matrices
    if not overall:
        print("\n[Comparison] FULL MATRICES:")
        print("[Comparison] Manual transformed data (ACTUAL):")
        print(manual_transformed)
        print("[Comparison] Library transformed data (EXPECTED):")
        print(library_transformed)
        print("\n[Comparison] Manual components (ACTUAL):")
        print(manual_components)
        print("[Comparison] Library components (EXPECTED):")
        print(library_components)
        print("\n[Comparison] Manual eigenvalues (ACTUAL):")
        print(manual_eigvals)
        print("[Comparison] Library eigenvalues (EXPECTED):")
        print(library_eigvals)
    
    return overall


def output_array_csv(arr, header=""):
    """
    Print a NumPy array as CSV rows to stdout.
    """
    if header:
        print(header)
    for row in arr:
        if np.ndim(row) == 0:
            # It's a scalar
            print(f"{row}")
        else:
            print(",".join(str(x) for x in row))


def output_array_human_readable(arr, name="Array"):
    """
    Print a NumPy array in a more human-readable format with row-by-row prints.
    """
    print(f"--- {name} (shape={arr.shape}) ---")
    if arr.ndim == 1:
        # Single vector
        print("[" + ", ".join([f"{v: .6f}" for v in arr]) + "]")
    else:
        for i, row in enumerate(arr):
            row_str = ", ".join([f"{v: .6f}" for v in row])
            print(f"Row {i}: {row_str}")
    print("-" * 40)


def run_random_test_suite(num_tests=5):
    """
    Run a set of random tests to thoroughly verify the manual vs library PCA
    under various shapes and n_components. Non-deterministic data ensures
    we test a variety of scenarios.
    """
    print("\n[Random Test-Suite] Beginning multiple random PCA tests...")
    for test_idx in range(1, num_tests + 1):
        # Randomly choose number of samples and features between [2..12]
        samples = random.randint(2, 12)
        features = random.randint(2, 12)

        # Possibly choose n_components up to the min dimension, or just None
        # 50% chance we use None, 50% chance we pick a random n_components
        if random.random() < 0.5:
            chosen_n_components = None
        else:
            chosen_n_components = random.randint(1, min(samples, features))

        print(f"\n[Test {test_idx}] samples={samples}, features={features}, "
              f"n_components={chosen_n_components if chosen_n_components else 'Auto'}")

        # Generate data (non-deterministically)
        X_rand = np.random.randn(samples, features)

        # Compare manual and library PCA
        result = compare_pca(X_rand, n_components=chosen_n_components)
        print(f"[Test {test_idx}] => PCA comparison result: {result}")

    print("[Random Test-Suite] All random tests completed.\n")

def run_controlled_structure_test():
    """
    Generate a large matrix with EXACTLY 3 real components and the rest pure noise.
    We know precisely which PCs should matter and which are just noise.
    """
    print("\n[Controlled Structure Test] Generating matrix with exactly 3 real components...")

    # Use same dimensions as the original genotype test
    n_samples = 88
    n_variants = 10000
    n_real_components = 3  # Exactly 3 components represent true structure
    signal_strength = [50, 20, 10]  # Stronger to weaker signals for each component

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create orthogonal basis vectors for the signal components
    # This ensures we get exactly n_real_components of signal
    random_basis = np.random.randn(n_samples, n_real_components)
    Q, _ = np.linalg.qr(random_basis)  # Orthogonalize
    true_factors = Q[:, :n_real_components]  # These are orthogonal unit vectors
    
    # Create loadings for each variant (n_real_components x n_variants)
    loadings = np.random.randn(n_real_components, n_variants)
    
    # Create the pure signal by combining factors and loadings, with decreasing strengths
    pure_signal = np.zeros((n_samples, n_variants))
    for i in range(n_real_components):
        component_signal = np.outer(true_factors[:, i], loadings[i, :])
        pure_signal += component_signal * signal_strength[i]
    
    # Add pure random noise
    noise = np.random.randn(n_samples, n_variants)
    
    # Final matrix = signal + noise
    X = pure_signal + noise
    
    print(f"[Controlled Test] Created data with {n_real_components} real components and pure noise")
    print(f"[Controlled Test] Matrix shape: {X.shape}")
    
    # Test PCA with more components than we know are real
    n_components = 5  # Request 5 components, but only 3 are "real"
    
    print("[Controlled Test] Now comparing manual vs library PCA...")
    result = compare_pca_with_ground_truth(X, n_components, n_real_components)
    print(f"[Controlled Test] => PCA meaningful structure match: {result}")
    return result

def compare_pca_with_ground_truth(X, n_components, n_real_components):
    """
    Compare PCA implementations, but only require high correlation for
    components we know represent real structure (not noise).
    """
    print("[Comparison] Comparing Manual PCA and Library PCA.")
    manual_transformed, manual_components, manual_eigvals = manual_pca(X, n_components)
    library_transformed, library_components, library_eigvals = library_pca(X, n_components)
    
    # Calculate and print explained variance
    total_variance_manual = np.sum(manual_eigvals)
    total_variance_library = np.sum(library_eigvals)
    
    print("\n[Comparison] Explained Variance:")
    print("Component | Manual PCA |  %  | Library PCA |  %  | Status")
    print("---------+------------+-----+-------------+-----+--------")
    for i in range(len(manual_eigvals)):
        manual_pct = (manual_eigvals[i] / total_variance_manual) * 100
        library_pct = (library_eigvals[i] / total_variance_library) * 100
        status = "REAL SIGNAL" if i < n_real_components else "PURE NOISE"
        print(f"    PC{i+1}  | {manual_eigvals[i]:10.2f} | {manual_pct:3.1f}% | {library_eigvals[i]:11.2f} | {library_pct:3.1f}% | {status}")
    
    print(f"\nTotal variance - Manual: {total_variance_manual:.2f}, Library: {total_variance_library:.2f}")
    print(f"First {n_real_components} PCs capture real structure, remaining PCs are pure noise")
    
    # Compare correlation for each component between implementations
    correlations = []
    print("\nComponent Correlations (accounting for sign flips):")
    print("Component | Correlation | Required | Status")
    print("---------+------------+----------+--------")
    
    all_match = True
    
    for i in range(n_components):
        manual_col = manual_transformed[:, i]
        library_col = library_transformed[:, i]
        
        # Calculate correlation (accounting for sign flips)
        corr_pos = np.abs(np.corrcoef(manual_col, library_col)[0, 1])
        corr_neg = np.abs(np.corrcoef(manual_col, -library_col)[0, 1])
        corr = max(corr_pos, corr_neg)
        correlations.append(corr)
        
        if i < n_real_components:
            # Real components must have strong correlation
            status = "MUST MATCH" 
            threshold = 0.95
            if corr < threshold:
                all_match = False
                result = "✗ FAILED"
            else:
                result = "✓ PASSED"
        else:
            # Noise components can differ completely
            status = "CAN DIFFER"
            threshold = 0.0
            result = "✓ IGNORED"
        
        print(f"    PC{i+1}  | {corr:10.4f} | >={threshold:.2f}     | {result}")
    
    return all_match

def main():
    print("=== PCA SCRIPT START ===\n")

    # For nicer numeric printing
    np.set_printoptions(precision=6, suppress=True)

    # Parse command-line arguments and possibly config
    args = parse_arguments()

    # Load config file if specified
    config = {}
    if args["config"]:
        config = parse_config_file(args["config"])

    # Merge config with command-line (command-line overrides)
    data_csv = args["data_csv"] or config.get("data_csv", None)
    n_components = args["n_components"] if args["n_components"] is not None else config.get("n_components", None)
    test_flag = args["test_flag"] or config.get("test_flag", False)
    samples = args["samples"] if args["samples"] is not None else config.get("samples", None)
    features = args["features"] if args["features"] is not None else config.get("features", None)
    random_seed = args["random_seed"] if args["random_seed"] is not None else config.get("random_seed", None)
    human_readable = args["human_readable"] or config.get("human_readable", False)

    # Acquire data either from CSV or by generation
    if data_csv:
        print("[Main] Loading data from CSV.")
        X = load_data_from_csv(data_csv)
    else:
        # If no CSV was provided, we'll just generate some data
        # for immediate usage (outside test-flag scenario).
        print("[Main] Generating random data (no CSV provided).")
        if samples is None:
            samples = 5
        if features is None:
            features = 5
        X = generate_random_data(samples, features, random_seed)

    # Handle test-flag
    if test_flag:
        # 1) If CSV was provided, do a direct compare on that data.
        if data_csv:
            print("[Main] --test-flag is set. Comparing manual vs library PCA on loaded CSV dataset.")
            compare_result = compare_pca(X, n_components)
            print(f"[Main] Single-dataset comparison result: {compare_result}")

        # 2) Always run the random test-suite to ensure broader coverage
        print("[Main] Now running random test-suite with non-deterministic data.")
        run_random_test_suite(num_tests=5)

        # 3)
        print("[Main] Also running controlled structure test with 3 real components.")
        run_controlled_structure_test()

    else:
        # No test-flag => run library PCA, output results
        print("[Main] --test-flag not set. Performing PCA (library) and outputting results.")
        transformed, components, eigvals = library_pca(X, n_components)

        # Output results
        if human_readable:
            print("\n[Output] PCA Results (Library) in human-readable form:")
            output_array_human_readable(transformed, name="Transformed Data")
            output_array_human_readable(components, name="Components (n_features x n_components)")
            output_array_human_readable(eigvals, name="Eigenvalues")
        else:
            print("\n[Output] PCA Results (Library) in CSV form:")
            print("Transformed Data (CSV):")
            output_array_csv(transformed)

            print("\nComponents (CSV):")
            output_array_csv(components)

            print("\nEigenvalues (CSV):")
            output_array_csv(eigvals.reshape(-1, 1))

    print("\n=== PCA SCRIPT END ===")

if __name__ == "__main__":
    main()
