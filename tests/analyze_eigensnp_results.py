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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
    backend_dirs = []
    try:
        backend_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        if not backend_dirs:
            logging.warning(f"No backend subdirectories found in {args.input_dir}")
            # Create an empty manifest and exit if no backends found
            with open(os.path.join(args.output_dir, "manifest.txt"), "w") as f:
                f.write("Backend\tRelativePath\tAbsolutePath\tSize\n")
            # Create empty summary results and report
            pd.DataFrame().to_csv(os.path.join(args.output_dir, "consolidated_summary_results.tsv"), sep='\t', index=False)
            with open(os.path.join(args.output_dir, "analysis_report.md"), "w") as md_file:
                md_file.write("# Analysis Report\n\nNo backend data found.\n")
            return 
    except FileNotFoundError:
        logging.error(f"Input directory {args.input_dir} not found.")
        return

    all_tsv_files_manifest = [] # For manifest.txt
    summary_dfs = [] # For consolidated_summary_results.tsv
    artifact_prefix = "eigensnp-test-artifacts-"

    logging.info(f"Scanning for backend data in subdirectories of: {args.input_dir}")

    for original_dir_name in backend_dirs:
        actual_backend_name = original_dir_name
        if original_dir_name.startswith(artifact_prefix):
            actual_backend_name = original_dir_name[len(artifact_prefix):]
            logging.info(f"Identified CI artifact directory: '{original_dir_name}'. Using backend name: '{actual_backend_name}'")
        else:
            logging.info(f"Processing directory: '{original_dir_name}'. Using backend name: '{actual_backend_name}' (no CI prefix detected)")

        # Path to the directory containing actual data (e.g. eigensnp-test-artifacts-openblas)
        backend_data_path = os.path.join(args.input_dir, original_dir_name)
        
        # Locate and parse eigensnp_summary_results.tsv
        # The summary file is expected to be directly inside the backend_data_path
        summary_file_path = os.path.join(backend_data_path, "eigensnp_summary_results.tsv")
        if os.path.exists(summary_file_path):
            try:
                summary_df = pd.read_csv(summary_file_path, sep='\t')
                summary_df['backend'] = actual_backend_name # Use extracted name
                summary_dfs.append(summary_df)
                
                # For the manifest, RelativePath should be relative to input_dir, 
                # but include the original directory name to maintain uniqueness and context.
                manifest_relative_path_for_summary = os.path.join(original_dir_name, "eigensnp_summary_results.tsv")
                all_tsv_files_manifest.append({
                    "Backend": actual_backend_name,
                    "RelativePath": manifest_relative_path_for_summary,
                    "AbsolutePath": os.path.abspath(summary_file_path),
                    "Size": os.path.getsize(summary_file_path)
                })
                logging.info(f"Successfully loaded summary for backend: {actual_backend_name} from {summary_file_path}")
            except Exception as e:
                logging.warning(f"Could not parse {summary_file_path} for backend {actual_backend_name}: {e}")
        else:
            logging.warning(f"eigensnp_summary_results.tsv not found for backend: {actual_backend_name} in {backend_data_path}")

        # Recursively find all other .tsv files within the backend_data_path
        # The glob pattern should search within the specific backend_data_path
        other_tsv_files = glob.glob(os.path.join(backend_data_path, "**", "*.tsv"), recursive=True)
        for tsv_file in other_tsv_files:
            # Avoid double-counting the summary file if it's caught by the glob
            if os.path.abspath(tsv_file) != os.path.abspath(summary_file_path):
                # RelativePath for manifest should be relative to input_dir, showing the full path from there
                relative_path_to_input_dir = os.path.relpath(tsv_file, args.input_dir)
                all_tsv_files_manifest.append({
                    "Backend": actual_backend_name, # Use extracted name
                    "RelativePath": relative_path_to_input_dir,
                    "AbsolutePath": os.path.abspath(tsv_file),
                    "Size": os.path.getsize(tsv_file)
                })

    # --- TSV Manifest ---
    manifest_df = pd.DataFrame(all_tsv_files_manifest)
    if not manifest_df.empty:
        manifest_path = os.path.join(args.output_dir, "manifest.txt")
        manifest_df.to_csv(manifest_path, sep='\t', index=False, columns=["Backend", "RelativePath", "AbsolutePath", "Size"])
        logging.info(f"Manifest file created at: {manifest_path}")
    else:
        logging.info("No TSV files found to create a manifest.")
        # Create an empty manifest if none found but backends existed
        with open(os.path.join(args.output_dir, "manifest.txt"), "w") as f:
            f.write("Backend\tRelativePath\tAbsolutePath\tSize\n")


    # --- Consolidation and Analysis of eigensnp_summary_results.tsv ---
    if not summary_dfs:
        logging.warning("No summary dataframes to consolidate.")
        # Create empty summary results and report if no summaries were loaded
        pd.DataFrame().to_csv(os.path.join(args.output_dir, "consolidated_summary_results.tsv"), sep='\t', index=False)
        with open(os.path.join(args.output_dir, "analysis_report.md"), "w") as md_file:
            md_file.write("# Analysis Report\n\nNo summary results found to analyze.\n")
        return # Stop further processing if no summaries

    consolidated_summary_df = pd.concat(summary_dfs, ignore_index=True)
    consolidated_summary_path = os.path.join(args.output_dir, "consolidated_summary_results.tsv")
    consolidated_summary_df.to_csv(consolidated_summary_path, sep='\t', index=False)
    logging.info(f"Consolidated summary results saved to: {consolidated_summary_path}")

    # --- Generate Markdown Report ---
    report_path = os.path.join(args.output_dir, "analysis_report.md")
    with open(report_path, "w") as md_file:
        md_file.write("# Eigensnp Test Analysis Report\n\n")
        
        # Overall summary
        md_file.write("## Overall Summary\n\n")
        total_tests_run = len(consolidated_summary_df)
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
            md_file.write(failed_tests_df[['TestName', 'backend', 'Success', 'NumPCsComputed', 'ErrorMessage']].to_markdown(index=False) + "\n\n")
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

        # --- Analysis of Specific Test TSVs (Example: Eigenvalues) ---
        md_file.write("## Eigenvalue Comparison Analysis\n\n")
        
        if manifest_df.empty:
            md_file.write("No TSV files were found in the manifest, so no eigenvalue comparison can be performed.\n\n")
        else:
            # Find files that might be eigenvalue files
            # A common pattern could be '*_eigenvalues.tsv' or 'eigenvalues.tsv'
            # We also need to group them by test scenario (e.g., parent directory)
            potential_eigenvalue_files = manifest_df[manifest_df['RelativePath'].str.contains("eigenvalues.tsv", case=False)]
            
            if potential_eigenvalue_files.empty:
                md_file.write("No files matching eigenvalue patterns (e.g., `*_eigenvalues.tsv`) found in the manifest.\n\n")
            else:
                # Extract test scenario from RelativePath (e.g., parent directory of the tsv file)
                # RelativePath in manifest_df is like `eigensnp-test-artifacts-openblas/test_scenario_dir/some_eigenvalues.tsv`
                # or `openblas/test_scenario_dir/some_eigenvalues.tsv`
                # The TestScenario should be the directory name containing the eigenvalues.tsv file.
                # We need to handle cases where RelativePath might have 2 or 3 components.
                # Example: original_dir_name/test_scenario/file.tsv -> test_scenario
                # Example: original_dir_name/file.tsv -> original_dir_name (as scenario)
                def extract_test_scenario(path_str):
                    parts = os.path.normpath(path_str).split(os.sep)
                    if len(parts) > 1:
                        return parts[-2] # Parent directory of the file
                    return parts[0] # The directory itself if the file is at the top level of its backend dir

                potential_eigenvalue_files['TestScenario'] = potential_eigenvalue_files['RelativePath'].apply(extract_test_scenario)
                
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
