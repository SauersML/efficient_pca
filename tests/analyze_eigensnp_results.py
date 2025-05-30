#!/usr/bin/env python3

import argparse
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Ensure no GUI backend is used
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path # Ensure Path is imported

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

import json # Added for JSON parsing

def parse_diagnostic_json(file_path):
    """Parses a single diagnostic JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully parsed diagnostic JSON: {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Diagnostic JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while parsing {file_path}: {e}")
        return None

def extract_key_metrics_from_diagnostics(parsed_json_data, diag_test_name, backend_name):
    """
    Extracts key metrics from parsed diagnostic JSON data.
    Returns a flat dictionary of these metrics.
    """
    metrics = {
        "diag_test_name": diag_test_name,
        "backend": backend_name,
        "overall_runtime_seconds": None,
        "local_block0_xsp_cond_num": None,
        "local_block0_avg_projb_cond_num": None,
        "local_block0_max_projb_cond_num": None,
        "local_block0_avg_q_ortho_error": None,
        "local_block0_max_q_ortho_error": None,
        "local_block0_up_avg_corr_vs_f64": None,
        "global_aeigen_cond_f32": None,
        # "global_aeigen_cond_f64": None, # This was in notes, harder to parse reliably
        "global_initial_scores_avg_corr": None,
        "sr_pass1_vqr_ortho_error": None,
        "sr_pass1_s_inter_cond_num": None,
    }

    if not parsed_json_data:
        return metrics

    try:
        metrics["overall_runtime_seconds"] = parsed_json_data.get("total_runtime_seconds")

        # Local basis diagnostics (assuming block 0 if it exists)
        per_block_diags = parsed_json_data.get("per_block_diagnostics", [])
        if per_block_diags and isinstance(per_block_diags, list) and len(per_block_diags) > 0:
            block0_diag = per_block_diags[0] # Taking the first block as representative
            if isinstance(block0_diag, dict):
                # This was 'u_p_condition_number' in the Rust struct, which was for X_s_p
                metrics["local_block0_xsp_cond_num"] = block0_diag.get("u_p_condition_number") 

                rsvd_stages = block0_diag.get("rsvd_stages", [])
                if isinstance(rsvd_stages, list):
                    projected_b_cond_nums = []
                    q_ortho_errors = []
                    for step in rsvd_stages:
                        if isinstance(step, dict):
                            if step.get("step_name") == "ProjectedB_PreSVD":
                                projected_b_cond_nums.append(step.get("condition_number"))
                            if step.get("step_name", "").endswith("_PostQR"): # Q0_PostQR, PowerIterX_Qtilde_PostQR, etc.
                                q_ortho_errors.append(step.get("orthogonality_error"))
                    
                    valid_projb_cond_nums = [x for x in projected_b_cond_nums if x is not None]
                    if valid_projb_cond_nums:
                        metrics["local_block0_avg_projb_cond_num"] = sum(valid_projb_cond_nums) / len(valid_projb_cond_nums)
                        metrics["local_block0_max_projb_cond_num"] = max(valid_projb_cond_nums)

                    valid_q_ortho_errors = [x for x in q_ortho_errors if x is not None]
                    if valid_q_ortho_errors:
                        metrics["local_block0_avg_q_ortho_error"] = sum(valid_q_ortho_errors) / len(valid_q_ortho_errors)
                        metrics["local_block0_max_q_ortho_error"] = max(valid_q_ortho_errors)

                u_corr = block0_diag.get("u_correlation_vs_f64_truth")
                if u_corr and isinstance(u_corr, list) and len(u_corr) > 0:
                    valid_u_corr = [x for x in u_corr if x is not None]
                    if valid_u_corr:
                        metrics["local_block0_up_avg_corr_vs_f64"] = sum(valid_u_corr) / len(valid_u_corr)
        
        # Global PCA diagnostics
        global_pca_diag = parsed_json_data.get("global_pca_diag")
        if isinstance(global_pca_diag, dict):
            global_rsvd_stages = global_pca_diag.get("rsvd_stages", [])
            if isinstance(global_rsvd_stages, list) and len(global_rsvd_stages) > 0:
                first_global_step = global_rsvd_stages[0] # Input_A_eigen_std
                if isinstance(first_global_step, dict):
                    metrics["global_aeigen_cond_f32"] = first_global_step.get("condition_number")
                    # global_aeigen_cond_f64 was in notes, harder to parse robustly here.

            initial_scores_corr = global_pca_diag.get("initial_scores_correlation_vs_py_truth")
            if initial_scores_corr and isinstance(initial_scores_corr, list) and len(initial_scores_corr) > 0:
                valid_is_corr = [x for x in initial_scores_corr if x is not None]
                if valid_is_corr:
                     metrics["global_initial_scores_avg_corr"] = sum(valid_is_corr) / len(valid_is_corr)

        # SR Pass Details (assuming first pass if it exists)
        sr_passes = parsed_json_data.get("sr_pass_details", [])
        if isinstance(sr_passes, list) and len(sr_passes) > 0:
            pass1_diag = sr_passes[0]
            if isinstance(pass1_diag, dict):
                # v_qr_ortho_error was originally planned, but SrPassDetail has v_hat_orthogonality_error and u_s_orthogonality_error.
                # u_s_orthogonality_error corresponds to the V_QR* from compute_refined_snp_loadings
                metrics["sr_pass1_vqr_ortho_error"] = pass1_diag.get("u_s_orthogonality_error") 
                metrics["sr_pass1_s_inter_cond_num"] = pass1_diag.get("s_intermediate_condition_number")


    except Exception as e:
        logging.error(f"Error extracting metrics for test {diag_test_name}, backend {backend_name}: {e}")
        # Keep already extracted metrics, others will remain None or default
    
    return metrics

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyzes eigensnp test results from various backends.")
    parser.add_argument("--input-dir", required=True, help="Directory containing backend-specific artifact subdirectories.")
    parser.add_argument("--output-dir", required=True, help="Directory where analysis files and plots will be saved.")
    return parser.parse_args()

def main():
    """Main function to orchestrate the analysis."""
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {args.output_dir}")

    # --- File Discovery ---
    artifact_prefix = "eigensnp-test-artifacts-"
    backend_dirs_to_process = [] # Store tuples of (original_dir_name, actual_backend_name)
    
    try:
        all_input_subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        for dir_name in all_input_subdirs:
            if dir_name.startswith(artifact_prefix):
                actual_backend_name = dir_name[len(artifact_prefix):]
                backend_dirs_to_process.append((dir_name, actual_backend_name))
                logging.info(f"Found CI artifact directory: '{dir_name}'. Extracted backend name: '{actual_backend_name}'")
            # else:
                # logging.info(f"Directory '{dir_name}' does not match prefix '{artifact_prefix}'. Skipping.")
                # Optionally, handle non-prefixed dirs for local runs if needed, or strictly process only prefixed ones.
                # For now, strictly processing prefixed ones as per typical CI artifact structure.
    except FileNotFoundError:
        logging.error(f"Input directory {args.input_dir} not found.")
        sys.exit(1)

    if not backend_dirs_to_process:
        logging.error(f"No backend artifact directories starting with '{artifact_prefix}' found in {args.input_dir}.")
        # Create empty outputs as per previous behavior, then exit
        with open(os.path.join(args.output_dir, "manifest.txt"), "w") as f: f.write("Backend\tRelativePath\tAbsolutePath\tSize\n")
        pd.DataFrame().to_csv(os.path.join(args.output_dir, "consolidated_summary_results.tsv"), sep='\t', index=False)
        with open(os.path.join(args.output_dir, "analysis_report.md"), "w") as md_file: md_file.write("# Analysis Report\n\nNo backend artifact directories found.\n")
        sys.exit(1)

    all_tsv_files_manifest = [] # For manifest.txt
    summary_dfs = [] # For consolidated_summary_results.tsv

    logging.info(f"Processing identified backend artifact directories: {[d[0] for d in backend_dirs_to_process]}")

    for original_dir_name, actual_backend_name in backend_dirs_to_process:
        # Revert base path to search directly in the backend artifact directory
        backend_artifacts_path = os.path.join(args.input_dir, original_dir_name)
        logging.info(f"Searching for artifacts in: {backend_artifacts_path} for backend: {actual_backend_name}")

        if not os.path.isdir(backend_artifacts_path):
            logging.warning(f"Artifact path {backend_artifacts_path} does not exist or is not a directory for backend {actual_backend_name}. Skipping.")
            continue
            
        summary_file_path = os.path.join(backend_artifacts_path, "eigensnp_summary_results.tsv")
        if os.path.exists(summary_file_path):
            try:
                summary_df = pd.read_csv(summary_file_path, sep='\t')
                if not summary_df.empty:
                    summary_df['backend'] = actual_backend_name
                    summary_dfs.append(summary_df)
                    
                    # RelativePath for manifest should be relative to input_dir, reflecting the full structure
                    manifest_relative_path = os.path.relpath(summary_file_path, args.input_dir)
                    all_tsv_files_manifest.append({
                        "Backend": actual_backend_name,
                        "RelativePath": manifest_relative_path,
                        "AbsolutePath": os.path.abspath(summary_file_path),
                        "Size": os.path.getsize(summary_file_path)
                    })
                    logging.info(f"Successfully loaded and processed non-empty summary for backend: {actual_backend_name} from {summary_file_path}")
                else:
                    logging.warning(f"Summary file {summary_file_path} for backend {actual_backend_name} is empty.")
            except Exception as e:
                logging.warning(f"Could not parse {summary_file_path} for backend {actual_backend_name}: {e}")
        else:
            logging.warning(f"eigensnp_summary_results.tsv not found for backend: {actual_backend_name} in {backend_artifacts_path}")

        # Recursively find all other .tsv files within the backend_artifacts_path
        other_tsv_files = glob.glob(os.path.join(backend_artifacts_path, "**", "*.tsv"), recursive=True)
        for tsv_file in other_tsv_files:
            if os.path.abspath(tsv_file) != os.path.abspath(summary_file_path): # Avoid double-counting
                relative_path_to_input_dir = os.path.relpath(tsv_file, args.input_dir)
                all_tsv_files_manifest.append({
                    "Backend": actual_backend_name,
                    "RelativePath": relative_path_to_input_dir,
                    "AbsolutePath": os.path.abspath(tsv_file),
                    "Size": os.path.getsize(tsv_file)
                })
    
    # --- TSV Manifest ---
    manifest_df = pd.DataFrame(all_tsv_files_manifest)
    if not manifest_df.empty:
        manifest_path = os.path.join(args.output_dir, "all_tsv_manifest.txt") # Renamed
        manifest_df.to_csv(manifest_path, sep='\t', index=False, columns=["Backend", "RelativePath", "AbsolutePath", "Size"])
        logging.info(f"Manifest file for all TSVs created at: {manifest_path}") # Updated log
    else:
        logging.info("No TSV files found to create a manifest.")
        # Create an empty manifest if none found but backends existed
        with open(os.path.join(args.output_dir, "all_tsv_manifest.txt"), "w") as f: # Renamed
            f.write("Backend\tRelativePath\tAbsolutePath\tSize\n")

    # --- Diagnostic JSON File Discovery ---
    all_diag_json_files_manifest = []
    logging.info("Starting discovery of diagnostic JSON files...")

    for original_dir_name, actual_backend_name in backend_dirs_to_process:
        # Path to where test artifacts for this backend are stored.
        # Based on CI structure, this is: args.input_dir / original_dir_name / "target" / "test_artifacts"
        # And diagnostic JSONs are in a "test_outputs" subdirectory within that.
        # So, backend_artifacts_path for glob should point to .../test_artifacts/
        
        # The structure of downloaded artifacts is:
        # args.input_dir / original_dir_name (e.g. "eigensnp-test-artifacts-openblas-diag") / ...
        # The actual test outputs (like JSON files) are usually in a nested `target/test_artifacts/test_outputs`
        # if the tests are run via `cargo test` which places artifacts relative to `target/`.
        
        # Construct the path to the directory where diagnostic JSONs are expected for this backend.
        # This matches where `run_diagnostic_test_with_params` saves them.
        # The `original_dir_name` is like `eigensnp-test-artifacts-openblas-diag`.
        # The JSONs are saved by tests into `test_outputs/` relative to where tests run.
        # In CI, artifacts are uploaded from `target/test_artifacts/`.
        # So, the path to search is `args.input_dir / original_dir_name / target/test_artifacts / test_outputs / diagnostics_*.json`
        
        current_backend_base_path = os.path.join(args.input_dir, original_dir_name)
        # The test helper saves into "test_outputs/" relative to where the test runs.
        # Cargo tests run from the package root. `target/test_artifacts` is where these are collected from.
        # So, the effective path within the downloaded artifact for `test_outputs` is `target/test_artifacts/test_outputs`.
        
        diag_json_search_path = os.path.join(current_backend_base_path, "target", "test_artifacts", "test_outputs")
        search_pattern = os.path.join(diag_json_search_path, "diagnostics_*.json")
        
        logging.info(f"Searching for diagnostic JSONs in: {diag_json_search_path} for backend {actual_backend_name} with pattern {search_pattern}")
        
        found_json_files = glob.glob(search_pattern, recursive=False) # JSONs are directly in test_outputs

        for json_file_path in found_json_files:
            absolute_path = os.path.abspath(json_file_path)
            relative_path = os.path.relpath(absolute_path, args.input_dir)
            file_size = os.path.getsize(absolute_path)
            
            # Extract DiagTestName from filename
            # Filename example: diagnostics_n100_d1000_b1_cpb5_k5_sr1_locIter0_globIter2_local0_global2.json
            # We want the suffix part like "local0_global2"
            filename_stem = os.path.splitext(os.path.basename(json_file_path))[0]
            parts = filename_stem.split('_')
            diag_test_name_from_file = "unknown_variant"
            # Try to find the suffix that indicates the test variant
            # This depends on the naming convention from `run_diagnostic_test_with_params`
            # Example suffix: "local0_global2", "local2_global2", "local4_global2"
            if len(parts) > 0:
                # The last part of the filename (before .json) is the "output_filename_suffix" from the test.
                diag_test_name_from_file = parts[-1]


            all_diag_json_files_manifest.append({
                "Backend": actual_backend_name,
                "DiagTestName": diag_test_name_from_file, 
                "RelativePath": relative_path,
                "AbsolutePath": absolute_path,
                "Size": file_size
            })
            logging.info(f"Found diagnostic JSON: {absolute_path} for backend {actual_backend_name}, test variant {diag_test_name_from_file}")

    # --- Diagnostic JSON Manifest ---
    diag_manifest_df = pd.DataFrame(all_diag_json_files_manifest)
    if not diag_manifest_df.empty:
        diag_manifest_path = os.path.join(args.output_dir, "diagnostic_json_manifest.tsv")
        # Ensure columns are in a consistent order
        diag_manifest_df = diag_manifest_df[["Backend", "DiagTestName", "RelativePath", "AbsolutePath", "Size"]]
        diag_manifest_df.to_csv(diag_manifest_path, sep='\t', index=False)
        logging.info(f"Diagnostic JSON manifest file created at: {diag_manifest_path}")
    else:
        logging.info("No diagnostic JSON files found to create a manifest.")
        # Create an empty manifest if none found
        with open(os.path.join(args.output_dir, "diagnostic_json_manifest.tsv"), "w") as f:
            f.write("Backend\tDiagTestName\tRelativePath\tAbsolutePath\tSize\n")

    # --- Parse Discovered Diagnostic JSON Files ---
    parsed_diag_data = []
    if not diag_manifest_df.empty:
        logging.info(f"Parsing {len(diag_manifest_df)} discovered diagnostic JSON files...")
        for index, row in diag_manifest_df.iterrows():
            abs_path = row['AbsolutePath']
            parsed_content = parse_diagnostic_json(abs_path)
            if parsed_content:
                # Store parsed data along with some manifest info for context
                parsed_diag_data.append({
                    "Backend": row['Backend'],
                    "DiagTestName": row['DiagTestName'],
                    "RelativePath": row['RelativePath'],
                    "ParsedData": parsed_content 
                })
        logging.info(f"Successfully parsed {len(parsed_diag_data)} out of {len(diag_manifest_df)} diagnostic JSON files.")
    else:
        logging.info("Diagnostic JSON manifest is empty. Nothing to parse.")

    # --- Extract Key Metrics from Parsed Diagnostic Data ---
    detailed_diagnostic_summaries = []
    if parsed_diag_data:
        logging.info(f"Extracting key metrics from {len(parsed_diag_data)} parsed diagnostic datasets...")
        for diag_item in parsed_diag_data:
            summary_dict = extract_key_metrics_from_diagnostics(
                diag_item["ParsedData"],
                diag_item["DiagTestName"],
                diag_item["Backend"]
            )
            detailed_diagnostic_summaries.append(summary_dict)
        logging.info(f"Successfully extracted metrics for {len(detailed_diagnostic_summaries)} diagnostic datasets.")

    if detailed_diagnostic_summaries:
        summary_diag_df = pd.DataFrame(detailed_diagnostic_summaries)
        summary_diag_path = os.path.join(args.output_dir, "consolidated_diagnostic_summary.tsv")
        summary_diag_df.to_csv(summary_diag_path, sep='\t', index=False)
        logging.info(f"Consolidated diagnostic summary saved to: {summary_diag_path}")
    else:
        logging.info("No detailed diagnostic summaries to save.")
        # Create an empty file if no summaries generated
        with open(os.path.join(args.output_dir, "consolidated_diagnostic_summary.tsv"), "w") as f:
            # Write header based on keys in 'metrics' dict from extract_key_metrics_from_diagnostics
            # This is a bit manual; could be more dynamic if needed
            header_keys = [
                "diag_test_name", "backend", "overall_runtime_seconds",
                "local_block0_xsp_cond_num", "local_block0_avg_projb_cond_num", "local_block0_max_projb_cond_num",
                "local_block0_avg_q_ortho_error", "local_block0_max_q_ortho_error", "local_block0_up_avg_corr_vs_f64",
                "global_aeigen_cond_f32", "global_initial_scores_avg_corr",
                "sr_pass1_vqr_ortho_error", "sr_pass1_s_inter_cond_num"
            ]
            f.write("\t".join(header_keys) + "\n")


    # --- Consolidation and Analysis of eigensnp_summary_results.tsv ---
    if not summary_dfs:
        logging.critical("Critical error: No non-empty eigensnp_summary_results.tsv files were found or loaded across all backends.")
        # Create empty outputs
        pd.DataFrame().to_csv(os.path.join(args.output_dir, "consolidated_summary_results.tsv"), sep='\t', index=False)
        with open(os.path.join(args.output_dir, "analysis_report.md"), "w") as md_file:
            md_file.write("# Analysis Report\n\nCritical error: No non-empty eigensnp_summary_results.tsv files found.\n")
        sys.exit(1)

    consolidated_summary_df = pd.concat(summary_dfs, ignore_index=True)
    
    if consolidated_summary_df.empty:
        logging.critical("Critical error: Consolidated summary data is empty. Cannot generate report.")
        # Ensure consolidated file is also empty or reflects this
        consolidated_summary_df.to_csv(os.path.join(args.output_dir, "consolidated_summary_results.tsv"), sep='\t', index=False)
        with open(os.path.join(args.output_dir, "analysis_report.md"), "w") as md_file:
            md_file.write("# Analysis Report\n\nCritical error: Consolidated summary data is empty.\n")
        sys.exit(1)
        
    consolidated_summary_path = os.path.join(args.output_dir, "consolidated_summary_results.tsv")
    consolidated_summary_df.to_csv(consolidated_summary_path, sep='\t', index=False)
    logging.info(f"Consolidated summary results saved to: {consolidated_summary_path}")

    # --- Generate Markdown Report ---
    report_path = os.path.join(args.output_dir, "analysis_report.md")
    with open(report_path, "w") as md_file:
        md_file.write("# Eigensnp Test Analysis Report\n\n")
        
        # Overall summary
        md_file.write("## Overall Summary\n\n")
        total_tests_run = len(consolidated_summary_df) # Should be > 0 if we passed the checks
        md_file.write(f"- Total test scenarios run (across all backends): {total_tests_run}\n")
        
        pass_fail_counts = consolidated_summary_df.groupby('backend')['Success'].agg(
            total= 'count',
            passed=lambda x: (x.astype(str).str.lower() == 'true').sum()
        ).reset_index()
        pass_fail_counts['failed'] = pass_fail_counts['total'] - pass_fail_counts['passed']

        md_file.write("### Pass/Fail Counts per Backend:\n")
        md_file.write(pass_fail_counts[['backend', 'passed', 'failed', 'total']].to_markdown(index=False) + "\n\n")

        # Table of failed tests
        # Ensure 'Success' column is string type for case-insensitive comparison
        failed_tests_df = consolidated_summary_df[consolidated_summary_df['Success'].astype(str).str.lower() != 'true']
        if not failed_tests_df.empty:
            md_file.write("## Failed Tests\n\n")
            columns_to_display = ['TestName', 'backend', 'Success', 'NumPCsComputed']
            if 'ErrorMessage' in failed_tests_df.columns:
                columns_to_display.append('ErrorMessage')
            md_file.write(failed_tests_df[columns_to_display].to_markdown(index=False) + "\n\n")
        else:
            md_file.write("## Failed Tests\n\nNo failed tests detected.\n\n")

        # Highlight discrepancies (Placeholder - needs more specific logic)
        md_file.write("## Discrepancy Analysis\n\n")
        # Example: NumPCsComputed variance for the same test
        # Group by TestName and check if NumPCsComputed is consistent
        discrepancies = consolidated_summary_df.groupby('TestName')['NumPCsComputed'].nunique()
        varying_npcs = discrepancies[discrepancies > 1].index.tolist()
        if varying_npcs:
            md_file.write("### Tests with varying NumPCsComputed across backends:\n")
            for test_name in varying_npcs:
                md_file.write(f"- **{test_name}**:\n")
                details = consolidated_summary_df[consolidated_summary_df['TestName'] == test_name][['backend', 'NumPCsComputed', 'Success']]
                md_file.write(details.to_markdown(index=False) + "\n")
            md_file.write("\n")
        else:
            md_file.write("No tests found with varying NumPCsComputed across backends.\n\n")
        
        # Further discrepancy: same test, different outcomes
        # Group by TestName and check unique Success values
        outcome_discrepancies = consolidated_summary_df.groupby('TestName')['Success'].apply(lambda x: x.astype(str).str.lower().unique())
        tests_with_mixed_outcomes = outcome_discrepancies[outcome_discrepancies.apply(len) > 1].index.tolist()

        if tests_with_mixed_outcomes:
            md_file.write("### Tests with different outcomes across backends:\n")
            for test_name in tests_with_mixed_outcomes:
                md_file.write(f"- **{test_name}**:\n")
                details = consolidated_summary_df[consolidated_summary_df['TestName'] == test_name][['backend', 'Success']]
                md_file.write(details.to_markdown(index=False) + "\n")
            md_file.write("\n")
        else:
            md_file.write("No tests found with different outcomes across backends.\n\n")


        # --- Plotting: Test Success per Backend ---
        if not pass_fail_counts.empty:
            plt.figure(figsize=(10, 6))
            # Plotting passed and failed bars side-by-side for each backend
            plot_df = pass_fail_counts.set_index('backend')[['passed', 'failed']]
            plot_df.plot(kind='bar', stacked=False) # Use stacked=False for side-by-side
            
            plt.title('Test Success/Failure Counts by Backend')
            plt.xlabel('Backend')
            plt.ylabel('Number of Tests')
            plt.xticks(rotation=45, ha="right")
            plt.legend(title='Outcome')
            plt.tight_layout()
            
            plot_path = os.path.join(args.output_dir, "test_success_by_backend.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Test success bar plot saved to: {plot_path}")
            md_file.write("## Test Success/Failure Plot\n\n")
            md_file.write(f"![Test Success by Backend]({os.path.basename(plot_path)})\n\n") # Relative path for markdown
        else:
            logging.warning("Not enough data to generate test success plot.")
            md_file.write("## Test Success/Failure Plot\n\nNot enough data to generate this plot.\n\n")
        
        # --- Diagnostic JSON Manifest in Report ---
        md_file.write("## Diagnostic JSON Files Found\n\n")
        if not diag_manifest_df.empty:
            # Display relevant columns for the report
            md_file.write(diag_manifest_df[['Backend', 'DiagTestName', 'RelativePath', 'Size']].to_markdown(index=False) + "\n\n")
        else:
            md_file.write("No diagnostic JSON files were found.\n\n")

        # --- Key Diagnostic Metrics Summary in Report ---
        md_file.write("## Key Diagnostic Metrics Summary\n\n")
        if 'summary_diag_df' in locals() and not summary_diag_df.empty:
            # Select key columns for the markdown report for brevity
            key_cols_for_report = [
                "diag_test_name", "backend", "overall_runtime_seconds",
                "local_block0_xsp_cond_num", 
                # "local_block0_avg_projb_cond_num", # Maybe too detailed for top summary
                "local_block0_up_avg_corr_vs_f64",
                "global_aeigen_cond_f32", 
                "global_initial_scores_avg_corr",
                # "sr_pass1_vqr_ortho_error", # Maybe too detailed
                "sr_pass1_s_inter_cond_num"
            ]
            # Ensure only existing columns are selected to avoid KeyErrors
            existing_key_cols = [col for col in key_cols_for_report if col in summary_diag_df.columns]
            
            md_file.write(summary_diag_df[existing_key_cols].to_markdown(index=False, floatfmt=".3e") + "\n\n")
        else:
            md_file.write("No detailed diagnostic metrics were extracted or available to summarize.\n\n")

        # --- Analysis of Specific Test TSVs (Example: Eigenvalues) ---
        md_file.write("## Eigenvalue Comparison Analysis (from all_tsv_manifest.txt)\n\n") # Clarified source
        
        if manifest_df.empty: # This refers to all_tsv_manifest.txt now
            md_file.write("No TSV files were found in the all_tsv_manifest.txt, so no eigenvalue comparison can be performed.\n\n")
        else:
            # Find files that might be eigenvalue files
            # A common pattern could be '*_eigenvalues.tsv' or 'eigenvalues.tsv'
            # We also need to group them by test scenario (e.g., parent directory)
            potential_eigenvalue_files = manifest_df[manifest_df['RelativePath'].str.contains("eigenvalues.tsv", case=False)]
            
            if potential_eigenvalue_files.empty:
                md_file.write("No files matching eigenvalue patterns (e.g., `*_eigenvalues.tsv`) found in the manifest.\n\n")
            else:
                # Extract test scenario from RelativePath (e.g., parent directory of the tsv file)
                # RelativePath in manifest_df will be like:
                # `eigensnp-test-artifacts-openblas/scenario_A/eigenvalues.tsv`
                # or `eigensnp-test-artifacts-openblas/top_level_results.tsv`
                def extract_test_scenario_from_manifest_path(path_str):
                    # path_str is relative to args.input_dir
                    # e.g., "eigensnp-test-artifacts-openblas/scenario_A/eigenvalues.tsv"
                    # parts: ["eigensnp-test-artifacts-openblas", "scenario_A", "eigenvalues.tsv"]
                    # e.g., "eigensnp-test-artifacts-openblas/top_level_results.tsv"
                    # parts: ["eigensnp-test-artifacts-openblas", "top_level_results.tsv"]
                    parts = os.path.normpath(path_str).split(os.sep)
                    
                    if len(parts) >= 3: # File is in a subdirectory of the backend artifact directory
                        # e.g., .../scenario_A/file.tsv -> scenario_A is parts[-2]
                        return parts[-2]
                    elif len(parts) == 2: # File is directly under the backend artifact directory
                        # e.g., .../top_level_results.tsv -> use backend artifact name (parts[0]) as scenario identifier
                        # or a special marker like "."
                        return parts[0] # This groups all top-level files under the backend's name
                    else:
                        logging.warning(f"Could not determine test scenario from path: {path_str} - path has too few components.")
                        return "unknown_scenario"

                potential_eigenvalue_files['TestScenario'] = potential_eigenvalue_files['RelativePath'].apply(extract_test_scenario_from_manifest_path)
                
                scenarios_with_eigenvalues = potential_eigenvalue_files.groupby('TestScenario')
                num_eigenvalue_plots = 0

                for scenario_name, group_df in scenarios_with_eigenvalues:
                    if group_df['Backend'].nunique() > 1: # Only compare if multiple backends have this file for the scenario
                        logging.info(f"Processing eigenvalue comparison for scenario: {scenario_name}")
                        plt.figure(figsize=(12, 7))
                        
                        file_details_for_report = []
                        
                        for idx, row in group_df.iterrows():
                            try:
                                eigen_df = pd.read_csv(row['AbsolutePath'], sep='\t', header=None)
                                if eigen_df.empty or eigen_df.shape[1] == 0:
                                    logging.warning(f"Eigenvalue file {row['AbsolutePath']} is empty or has no columns.")
                                    continue
                                
                                eigenvalues = eigen_df.iloc[:, 0].dropna().astype(float)
                                if eigenvalues.empty:
                                    logging.warning(f"No numeric data in the first column of {row['AbsolutePath']}.")
                                    continue

                                # Take top N eigenvalues, e.g., 10 or min(10, len(eigenvalues))
                                top_n = min(10, len(eigenvalues))
                                plt.plot(range(1, top_n + 1), eigenvalues.head(top_n), marker='o', linestyle='-', label=f"{row['Backend']} (Top {top_n})")
                                file_details_for_report.append({
                                    'Backend': row['Backend'], 
                                    'File': os.path.join(scenario_name, os.path.basename(row['RelativePath'])),
                                    'NumEigenvalues': len(eigenvalues)
                                })
                            except Exception as e:
                                logging.error(f"Error processing eigenvalue file {row['AbsolutePath']} for scenario {scenario_name}: {e}")
                        
                        if not plt.gca().lines: # Check if any lines were actually plotted
                            logging.info(f"No valid eigenvalue data plotted for scenario: {scenario_name}. Skipping plot generation.")
                            plt.close() # Close the empty figure
                            continue

                        plt.title(f'Top Eigenvalue Comparison: {scenario_name}')
                        plt.xlabel('Component Index')
                        plt.ylabel('Eigenvalue')
                        plt.xticks(range(1, top_n + 1) if 'top_n' in locals() and top_n >0 else []) # ensure x-ticks are integers
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        plot_filename = f"eigenvalue_comparison_{scenario_name.replace(' ', '_').replace('/', '_')}.png"
                        plot_path = os.path.join(args.output_dir, plot_filename)
                        plt.savefig(plot_path)
                        plt.close()
                        logging.info(f"Eigenvalue comparison plot saved to: {plot_path} for scenario: {scenario_name}")
                        
                        md_file.write(f"### Scenario: {scenario_name}\n\n")
                        md_file.write(f"![Eigenvalue Comparison for {scenario_name}]({plot_filename})\n\n")
                        md_file.write("Files used in this comparison:\n")
                        md_file.write(pd.DataFrame(file_details_for_report).to_markdown(index=False) + "\n\n")
                        num_eigenvalue_plots +=1
                    else:
                        logging.info(f"Skipping eigenvalue comparison for scenario '{scenario_name}' as it does not involve multiple backends.")

                if num_eigenvalue_plots == 0:
                    md_file.write("No eigenvalue comparisons were generated. This could be because no test scenarios with eigenvalue files were found across multiple backends, or the files were empty/invalid.\n\n")
        
    logging.info("Analysis script completed.")

if __name__ == "__main__":
    main()