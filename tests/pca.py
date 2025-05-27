import argparse
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json # For original_main_logic
import os   # For original_main_logic
import scipy.linalg as la # For original_main_logic
import random # For original_main_logic


def print_numpy_array_for_rust(arr):
    """Prints a NumPy array in a space-separated format for Rust parsing."""
    if arr.size == 0:
        # For empty arrays (e.g. 0 components, or 0 SNPs/samples resulting in 0-size arrays)
        # Rust parser expects an empty line if the section itself is present but data is empty.
        # However, if the array is, e.g., (0, M) or (N, 0), it's better to print nothing for that array.
        # The section headers will still be printed.
        # Let's refine: if a 2D array has a zero dimension, print a newline.
        # If a 1D array is empty, print a newline.
        if arr.ndim == 0: # Should not happen for our cases (loadings, scores, eigenvalues)
            print("")
        elif arr.ndim == 1 and arr.shape[0] == 0: # Empty 1D array
            print("")
        elif arr.ndim == 2 and (arr.shape[0] == 0 or arr.shape[1] == 0): # Empty 2D array
            print("")
        else: # Non-empty array but somehow arr.size was 0? This case is unlikely if handled above.
             pass # No data to print, but section header was printed.
        return

    if arr.ndim == 1:
        # Print each element of the 1D array on a new line
        for element in arr:
            print(str(element))
    else: # 2D
        for row in arr:
            print(" ".join(map(str, row)))

def run_reference_pca_mode(n_components_val): # Renamed from run_rust_test_mode
    """Reads data from stdin, performs PCA, and prints results for Rust tests."""
    try:
        lines = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                lines.append(list(map(float, line.split())))
        
        # Initialize with empty arrays of appropriate dimensionality for the case of no input data
        # Loadings: k_eff x D (num_snps_d)
        # Scores: N (num_samples_n) x k_eff
        # Eigenvalues: k_eff
        num_snps_d_fallback = 0 # Fallback if no lines are read
        num_samples_n_fallback = 0

        if not lines:
            data_snps_by_samples = np.array([]).reshape(0,0)
        else:
            data_snps_by_samples = np.array(lines)
            num_snps_d_fallback = data_snps_by_samples.shape[0] if data_snps_by_samples.ndim == 2 else 0
            num_samples_n_fallback = data_snps_by_samples.shape[1] if data_snps_by_samples.ndim == 2 else 0


        if data_snps_by_samples.size == 0:
            # This handles truly empty input or input that parses to an empty array
            print("LOADINGS:")
            print_numpy_array_for_rust(np.array([]).reshape(0, num_snps_d_fallback))
            print("SCORES:")
            print_numpy_array_for_rust(np.array([]).reshape(num_samples_n_fallback, 0))
            print("EIGENVALUES:")
            print_numpy_array_for_rust(np.array([]))
            return

        num_snps_d = data_snps_by_samples.shape[0]
        num_samples_n = data_snps_by_samples.shape[1]

        if num_snps_d == 0 or num_samples_n == 0:
            k_eff = 0 
            print("LOADINGS:")
            print_numpy_array_for_rust(np.array([]).reshape(k_eff, num_snps_d))
            print("SCORES:")
            print_numpy_array_for_rust(np.array([]).reshape(num_samples_n, k_eff))
            print("EIGENVALUES:")
            print_numpy_array_for_rust(np.array([]))
            return

        data_samples_by_snps = data_snps_by_samples.T
        
        scaler = StandardScaler(with_mean=True, with_std=True)
        data_standardized = scaler.fit_transform(data_samples_by_snps)

        effective_n_components = min(n_components_val, num_samples_n, num_snps_d)
        if effective_n_components <= 0 : 
            print("LOADINGS:")
            print_numpy_array_for_rust(np.array([]).reshape(0, num_snps_d))
            print("SCORES:")
            print_numpy_array_for_rust(np.array([]).reshape(num_samples_n, 0))
            print("EIGENVALUES:")
            print_numpy_array_for_rust(np.array([]))
            return

        pca = PCA(n_components=effective_n_components, svd_solver='full')
        scores = pca.fit_transform(data_standardized)
        loadings = pca.components_
        eigenvalues = pca.explained_variance_

        print("LOADINGS:")
        print_numpy_array_for_rust(loadings)
        print("SCORES:")
        print_numpy_array_for_rust(scores)
        print("EIGENVALUES:")
        print_numpy_array_for_rust(eigenvalues)

    except ValueError as e:
        print(f"Error in run_reference_pca_mode: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in run_reference_pca_mode: {e}", file=sys.stderr)
        sys.exit(1)


# --- Functions for original script logic ---
def parse_config_file_original(config_path):
    if not os.path.isfile(config_path): return {}
    with open(config_path, 'r') as f: config = json.load(f)
    return config

def load_data_from_csv_original(csv_path):
    return np.loadtxt(csv_path, delimiter=",")

def generate_random_data_original(samples=5, features=5, random_seed=None):
    if random_seed is not None: np.random.seed(random_seed)
    return np.random.randn(samples, features)

def manual_pca_original(X, n_components=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape
    if n_components is None: n_components = min(n_samples, n_features)
    else: n_components = min(n_components, min(n_samples, n_features))
    
    if n_components <= 0: # Handle case where effective components is 0 or less
        return np.zeros((n_samples, 0)), np.zeros((n_features, 0)), np.array([])


    if n_features > n_samples:
        gram_matrix = np.dot(X_scaled, X_scaled.T) / (n_samples - 1)
        eigvals, eigvecs = la.eigh(gram_matrix)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:n_components]
        eigvecs = eigvecs[:, idx][:, :n_components]
        components = np.zeros((n_features, n_components))
        for i in range(n_components):
            val = eigvals[i]
            if val < 1e-12: components[:, i] = 0.0
            else:
                scale_factor = np.sqrt(val)
                components[:, i] = (X_scaled.T @ eigvecs[:, i]) / (scale_factor * np.sqrt(n_samples - 1))
                comp_norm = np.linalg.norm(components[:, i])
                if comp_norm > 1e-12: components[:, i] /= comp_norm
        X_transformed = X_scaled @ components
    else:
        cov_matrix = (X_scaled.T @ X_scaled) / (n_samples - 1)
        eigvals, eigvecs = la.eigh(cov_matrix)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:n_components]
        components = eigvecs[:, idx][:, :n_components]
        X_transformed = X_scaled @ components
    return X_transformed, components, eigvals

def library_pca_original(X, n_components=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape
    max_components = min(n_samples, n_features)
    if n_components is None: n_components = max_components
    else: n_components = min(n_components, max_components)
    
    if n_components <=0 : 
        return np.zeros((n_samples, 0)), np.zeros((n_features, 0)), np.array([])

    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)
    components = pca.components_.T
    explained_variance = pca.explained_variance_
    return X_transformed, components, explained_variance

def compare_pca_original(X, n_components=None):
    manual_transformed, _, manual_eigvals = manual_pca_original(X, n_components)
    library_transformed, _, library_eigvals = library_pca_original(X, n_components)
    transformed_similar = True
    if manual_transformed.shape != library_transformed.shape : return False 
    if manual_transformed.size > 0 : 
        for i in range(manual_transformed.shape[1]):
            manual_col = manual_transformed[:, i]
            library_col = library_transformed[:, i]
            sim_pos = np.allclose(manual_col, library_col, rtol=1e-5, atol=1e-5)
            sim_neg = np.allclose(manual_col, -library_col, rtol=1e-5, atol=1e-5)
            if not (sim_pos or sim_neg): transformed_similar = False; break
    eigvals_similar = np.allclose(manual_eigvals, library_eigvals, rtol=1e-5, atol=1e-5) if manual_eigvals.size > 0 or library_eigvals.size > 0 else True
    return transformed_similar and eigvals_similar

def output_array_csv_original(arr, header=""):
    if header: print(header)
    for row in arr:
        if np.ndim(row) == 0: print(f"{row}")
        else: print(",".join(str(x) for x in row))

def output_array_human_readable_original(arr, name="Array"):
    print(f"--- {name} (shape={arr.shape}) ---")
    if arr.ndim == 1: print("[" + ", ".join([f"{v: .6f}" for v in arr]) + "]")
    else:
        for i, row in enumerate(arr): print(f"Row {i}: {", ".join([f'{v: .6f}' for v in row])}")
    print("-" * 40)

def run_random_test_suite_original(num_tests=5):
    for test_idx in range(1, num_tests + 1):
        samples = random.randint(2, 12); features = random.randint(2, 12)
        chosen_n_components = None if random.random() < 0.5 else random.randint(1, min(samples, features))
        X_rand = np.random.randn(samples, features)
        result = compare_pca_original(X_rand, n_components=chosen_n_components)
        print(f"[Test {test_idx}] samples={samples}, features={features}, n_components={chosen_n_components if chosen_n_components else 'Auto'} => Result: {result}")

def original_main_logic(args):
    np.set_printoptions(precision=6, suppress=True)
    config = {}
    if args.config: config = parse_config_file_original(args.config)

    data_csv = args.data_csv if args.data_csv is not None else config.get("data_csv")
    n_components_orig = args.n_components if args.n_components is not None else config.get("n_components")
    test_flag = args.test_flag # Directly use the boolean value
    samples = args.samples if args.samples is not None else config.get("samples")
    features = args.features if args.features is not None else config.get("features")
    random_seed = args.random_seed if args.random_seed is not None else config.get("random_seed")
    human_readable = args.human_readable # Directly use the boolean value


    if data_csv: X = load_data_from_csv_original(data_csv)
    else:
        samples = samples if samples is not None else 5
        features = features if features is not None else 5
        X = generate_random_data_original(samples, features, random_seed)

    if test_flag:
        if data_csv:
            compare_result = compare_pca_original(X, n_components_orig)
            print(f"Single-dataset comparison result: {compare_result}")
        run_random_test_suite_original()
    else:
        transformed, components, eigvals = library_pca_original(X, n_components_orig)
        if human_readable:
            output_array_human_readable_original(transformed, name="Transformed Data")
            output_array_human_readable_original(components, name="Components (n_features x n_components)")
            output_array_human_readable_original(eigvals, name="Eigenvalues")
        else:
            output_array_csv_original(transformed, header="Transformed Data (CSV):")
            output_array_csv_original(components, header="\nComponents (CSV):")
            output_array_csv_original(eigvals.reshape(-1, 1), header="\nEigenvalues (CSV):")


def parse_arguments_main():
    """Parse all command-line arguments for main operation modes."""
    parser = argparse.ArgumentParser(description="PCA Script supporting general use and Rust test reference generation.")
    
    # Arguments for original mode (mutually exclusive group with --generate-reference-pca might be too strict if -k is shared)
    # These are relevant if --generate-reference-pca is NOT set
    parser.add_argument("--config", type=str, help="Path to a JSON config file for original mode.")
    parser.add_argument("--data_csv", type=str, help="Path to input CSV (samples x features) for original mode.")
    parser.add_argument("--samples", type=int, help="Num samples for random data generation in original mode.")
    parser.add_argument("--features", type=int, help="Num features for random data generation in original mode.")
    parser.add_argument("--random_seed", type=int, help="Random seed for original mode data generation.")
    parser.add_argument("--test-flag", action="store_true", help="Compare manual vs library PCA in original mode.")
    parser.add_argument("--human-readable", action="store_true", help="Human-readable output for original mode.")
    
    # Argument for n_components, used by both modes
    # For --generate-reference-pca, it's required. For original_main_logic, it's optional.
    parser.add_argument("-k", "--n-components", type=int, help="Number of components for PCA.")
    
    # Argument to switch to reference generation mode
    parser.add_argument("--generate-reference-pca", 
                        action="store_true", 
                        help="Enable mode for generating PCA reference results for Rust tests. Expects data via stdin and -k for n_components.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments_main()

    if args.generate_reference_pca:
        if args.n_components is None:
            print("Error: --n-components (-k) is required for --generate-reference-pca mode.", file=sys.stderr)
            sys.exit(1)
        run_reference_pca_mode(args.n_components) # Renamed function
    else:
        original_main_logic(args)
