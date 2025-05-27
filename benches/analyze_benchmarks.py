import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import glob
import os
import re
import numpy as np # For NaN

# Define output directory
OUTPUT_DIR = "benches/output"
ARTIFACTS_DIR = "benches/benchmark_artifacts"
KEY_SCENARIOS_FOR_PLOTS_STATS = ["Large", "Wide", "Sparse-W"]

def load_data(artifacts_path):
    """
    Reads all *.tsv files in artifacts_path.
    Extracts backend name from filename if not in 'BackendName' column.
    Concatenates into a single DataFrame.
    """
    all_files = glob.glob(os.path.join(artifacts_path, "*.tsv"))
    if not all_files:
        print(f"Warning: No TSV files found in {artifacts_path}")
        return pd.DataFrame()

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, sep='\t')
            # BackendName is already in the TSV, so direct extraction from filename is a fallback.
            if 'BackendName' not in df.columns:
                # Extract backend from filename, e.g., raw-benchmark-results-mkl.tsv -> mkl
                match = re.search(r'raw-benchmark-results-(.*?)\.tsv', os.path.basename(f))
                if match:
                    df['BackendName'] = match.group(1)
                else:
                    df['BackendName'] = 'unknown'
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: File {f} is empty and will be skipped.")
        except Exception as e:
            print(f"Warning: Could not read file {f} due to error: {e}")
    
    if not df_list:
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def clean_data(df):
    """
    Converts columns to appropriate types.
    Handles 'None' in NumComponentsOverride.
    """
    if df.empty:
        return df

    for col in ['TimeSec', 'RSSDeltaKB', 'VirtDeltaKB']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'NumSamples' in df.columns:
        df['NumSamples'] = pd.to_numeric(df['NumSamples'], errors='coerce')
    if 'NumFeatures' in df.columns:
        df['NumFeatures'] = pd.to_numeric(df['NumFeatures'], errors='coerce')
    if 'Iteration' in df.columns: # Assuming 'Iteration' is 'iteration_idx'
        df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')


    if 'NumComponentsOverride' in df.columns:
        # Convert "None" string to np.nan for numeric consistency if desired, or keep as string 'None'
        # For grouping, string 'None' is fine. If it needs to be numeric for some calculation, use NaN.
        df['NumComponentsOverride'] = df['NumComponentsOverride'].replace('None', np.nan)
        # Attempt to convert to numeric, but allow non-numeric by coercing errors to NaN
        df['NumComponentsOverride'] = pd.to_numeric(df['NumComponentsOverride'], errors='coerce')


    # Ensure categorical columns are strings for consistent grouping
    for col in ['ScenarioName', 'BackendName', 'RunType']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df

def aggregate_data(df):
    """
    Calculates mean, std for time and memory metrics per group.
    """
    if df.empty:
        return pd.DataFrame()

    group_by_cols = [
        'ScenarioName', 'NumSamples', 'NumFeatures', 
        'BackendName', 'RunType', 'NumComponentsOverride'
    ]
    
    # Filter out rows where any group_by_col is NaN if that's desired, or they'll be grouped as NaN
    # For NumComponentsOverride, NaN is a valid group after cleaning.

    # Check if all group_by_cols are present
    missing_cols = [col for col in group_by_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for aggregation: {missing_cols}. Skipping aggregation.")
        return pd.DataFrame()

    grouped = df.groupby(group_by_cols, dropna=False) # dropna=False to include NaN groups for NumComponentsOverride
    
    aggregated_df = grouped[['TimeSec', 'RSSDeltaKB', 'VirtDeltaKB']].agg(['mean', 'std']).reset_index()
    
    # Flatten MultiIndex columns
    new_cols = []
    for col_top, col_stat in aggregated_df.columns:
        if col_stat: # For 'mean' and 'std'
            new_cols.append(f"{col_top}_{col_stat}")
        else: # For group_by_cols
            new_cols.append(col_top)
    aggregated_df.columns = new_cols
    
    return aggregated_df

def generate_plots(df_raw):
    """
    Generates and saves box plots for key scenarios.
    """
    if df_raw.empty:
        print("Warning: Raw data is empty. Skipping plot generation.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for scenario in KEY_SCENARIOS_FOR_PLOTS_STATS:
        for run_type in ["fit", "rfit"]:
            scenario_df = df_raw[(df_raw['ScenarioName'] == scenario) & (df_raw['RunType'] == run_type)]
            if scenario_df.empty:
                print(f"Warning: No data for scenario {scenario}, run_type {run_type}. Skipping plots.")
                continue

            # TimeSec plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='BackendName', y='TimeSec', data=scenario_df)
            plt.title(f'Time Comparison for {scenario} - {run_type}')
            plt.ylabel('Time (seconds)')
            plt.xlabel('Backend')
            plot_filename_time = os.path.join(OUTPUT_DIR, f'plot_time_{scenario}_{run_type}.png')
            try:
                plt.savefig(plot_filename_time)
                print(f"Saved plot: {plot_filename_time}")
            except Exception as e:
                print(f"Error saving plot {plot_filename_time}: {e}")
            plt.close()

            # RSSDeltaKB plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='BackendName', y='RSSDeltaKB', data=scenario_df)
            plt.title(f'RSS Memory Comparison for {scenario} - {run_type}')
            plt.ylabel('RSS Delta (KB)')
            plt.xlabel('Backend')
            plot_filename_rss = os.path.join(OUTPUT_DIR, f'plot_rss_{scenario}_{run_type}.png')
            try:
                plt.savefig(plot_filename_rss)
                print(f"Saved plot: {plot_filename_rss}")
            except Exception as e:
                print(f"Error saving plot {plot_filename_rss}: {e}")
            plt.close()

def perform_statistical_tests(df_raw):
    """
    Performs ANOVA/Kruskal-Wallis and post-hoc tests for key scenarios.
    Returns a DataFrame or dict with results. (Placeholder for now)
    """
    if df_raw.empty:
        print("Warning: Raw data is empty. Skipping statistical tests.")
        return pd.DataFrame()
        
    results_list = []

    for scenario in KEY_SCENARIOS_FOR_PLOTS_STATS:
        for run_type in ["fit", "rfit"]:
            for metric in ['TimeSec', 'RSSDeltaKB']: # Add VirtDeltaKB if needed
                scenario_run_metric_df = df_raw[
                    (df_raw['ScenarioName'] == scenario) & 
                    (df_raw['RunType'] == run_type) &
                    df_raw[metric].notna() # Ensure metric data is not NaN
                ]
                
                if scenario_run_metric_df.empty:
                    # print(f"No data for stats: {scenario}, {run_type}, {metric}")
                    continue

                backends = scenario_run_metric_df['BackendName'].unique()
                if len(backends) < 2:
                    # print(f"Not enough backends for comparison: {scenario}, {run_type}, {metric} (backends: {backends})")
                    continue

                # Collect data for each backend
                grouped_data = [
                    scenario_run_metric_df[scenario_run_metric_df['BackendName'] == backend][metric]
                    for backend in backends
                ]
                
                # Check if all groups have sufficient data (e.g., >1 for Kruskal-Wallis)
                if any(len(group) < 2 for group in grouped_data): # Kruskal-Wallis needs at least 2 samples in each group if comparing >2 groups
                    # print(f"Skipping {scenario}/{run_type}/{metric} due to insufficient data in one or more backend groups.")
                    continue


                # Kruskal-Wallis test (non-parametric ANOVA alternative)
                try:
                    h_stat, p_value = stats.kruskal(*grouped_data)
                    results_list.append({
                        'ScenarioName': scenario,
                        'RunType': run_type,
                        'Metric': metric,
                        'Test': 'Kruskal-Wallis',
                        'H-Statistic': h_stat,
                        'P-Value': p_value,
                        'Significant_Overall': p_value < 0.05 # Example significance level
                    })
                    # Post-hoc tests (e.g., Dunn's test) would go here if p_value is significant
                except ValueError as e:
                    print(f"Error during Kruskal-Wallis for {scenario}/{run_type}/{metric}: {e}")
                    results_list.append({
                        'ScenarioName': scenario,
                        'RunType': run_type,
                        'Metric': metric,
                        'Test': 'Kruskal-Wallis',
                        'Error': str(e)
                    })


    if not results_list:
        return pd.DataFrame()
    return pd.DataFrame(results_list)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR) # Create if it doesn't exist, useful for local runs
        print(f"Created artifacts directory (for local testing): {ARTIFACTS_DIR}")
        # In a real run, files are expected here. For a first run, it might be empty.
        # Consider adding dummy files to ARTIFACTS_DIR for initial testing if needed.

    raw_df = load_data(ARTIFACTS_DIR)
    if raw_df.empty:
        print(f"No data loaded from {ARTIFACTS_DIR}. Exiting analysis script.")
        # Create empty placeholder files to satisfy GitHub Actions if it expects them
        placeholder_agg_file = os.path.join(OUTPUT_DIR, "consolidated_benchmark_analysis.tsv")
        placeholder_stats_file = os.path.join(OUTPUT_DIR, "statistical_analysis_summary.tsv")
        if not os.path.exists(placeholder_agg_file):
             pd.DataFrame().to_csv(placeholder_agg_file, sep='\t', index=False)
             print(f"Created empty placeholder: {placeholder_agg_file}")
        if not os.path.exists(placeholder_stats_file):
             pd.DataFrame().to_csv(placeholder_stats_file, sep='\t', index=False)
             print(f"Created empty placeholder: {placeholder_stats_file}")
        return

    raw_df = clean_data(raw_df)
    
    aggregated_df = aggregate_data(raw_df.copy()) # Use .copy() if clean_data modifies df and you need original raw_df elsewhere

    if not aggregated_df.empty:
        agg_file_path = os.path.join(OUTPUT_DIR, "consolidated_benchmark_analysis.tsv")
        aggregated_df.to_csv(agg_file_path, sep='\t', index=False, float_format='%.6f')
        print(f"Consolidated analysis TSV saved to: {agg_file_path}")
    else:
        print("Aggregated data is empty. Skipping saving consolidated_benchmark_analysis.tsv.")
        # Create empty placeholder if it doesn't exist
        placeholder_agg_file = os.path.join(OUTPUT_DIR, "consolidated_benchmark_analysis.tsv")
        if not os.path.exists(placeholder_agg_file):
             pd.DataFrame().to_csv(placeholder_agg_file, sep='\t', index=False)
             print(f"Created empty placeholder: {placeholder_agg_file}")


    generate_plots(raw_df) 
    stats_results_df = perform_statistical_tests(raw_df)

    print("\nStatistical Test Summary:")
    if stats_results_df is not None and not stats_results_df.empty:
        print(stats_results_df.to_string())
        stats_file_path = os.path.join(OUTPUT_DIR, "statistical_analysis_summary.tsv")
        stats_results_df.to_csv(stats_file_path, sep='\t', index=False, float_format='%.6f')
        print(f"Statistical analysis summary saved to: {stats_file_path}")
    else:
        print("No statistical tests performed or no results generated.")
        # Create empty placeholder if it doesn't exist
        placeholder_stats_file = os.path.join(OUTPUT_DIR, "statistical_analysis_summary.tsv")
        if not os.path.exists(placeholder_stats_file):
             pd.DataFrame().to_csv(placeholder_stats_file, sep='\t', index=False)
             print(f"Created empty placeholder: {placeholder_stats_file}")


if __name__ == "__main__":
    main()
