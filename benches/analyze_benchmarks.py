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
# KEY_SCENARIOS_FOR_PLOTS_STATS is used by Kruskal-Wallis, but plots and M-W tests run on all scenarios.
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
            if 'BackendName' not in df.columns:
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

    for col_num in ['NumSamples', 'NumFeatures', 'Iteration']:
        if col_num in df.columns: # iteration_idx is Iteration in some files
            df[col_num] = pd.to_numeric(df[col_num], errors='coerce')
            if col_num == 'Iteration' and 'iteration_idx' not in df.columns: # Make sure it's consistent
                 df['iteration_idx'] = df[col_num]


    if 'NumComponentsOverride' in df.columns:
        df['NumComponentsOverride'] = df['NumComponentsOverride'].replace('None', np.nan)
        df['NumComponentsOverride'] = pd.to_numeric(df['NumComponentsOverride'], errors='coerce')

    for col_cat in ['ScenarioName', 'BackendName', 'RunType']:
        if col_cat in df.columns:
            df[col_cat] = df[col_cat].astype(str)
            
    return df

def aggregate_data(df):
    """
    Calculates mean, std for time and memory metrics per group.
    """
    if df.empty:
        return pd.DataFrame()

    group_by_cols = ['ScenarioName', 'NumSamples', 'NumFeatures', 'BackendName', 'RunType', 'NumComponentsOverride']
    missing_cols = [col for col in group_by_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for aggregation: {missing_cols}. Skipping aggregation.")
        return pd.DataFrame()

    # Ensure NumComponentsOverride is treated as a string for grouping if it contains NaNs that should be distinct group
    # However, previous cleaning converts it to numeric or NaN. For grouping, NaN is a valid group key.
    # If you want 'None' (string) and NaN (numeric) to be treated as separate groups, ensure type consistency before grouping.
    # Here, we assume NaNs in NumComponentsOverride are to be grouped together if dropna=False.
    
    grouped = df.groupby(group_by_cols, dropna=False) 
    aggregated_df = grouped[['TimeSec', 'RSSDeltaKB', 'VirtDeltaKB']].agg(['mean', 'std']).reset_index()
    
    new_cols = []
    for col_top, col_stat in aggregated_df.columns:
        new_cols.append(f"{col_top}_{col_stat}" if col_stat else col_top)
    aggregated_df.columns = new_cols
    
    return aggregated_df

def format_p_value(p_value):
    """Formats p-value for display on plots."""
    if pd.isna(p_value):
        return "n/a" # Should indicate that test was not performed or error
    if p_value < 0.001:
        return "p < 0.001***"
    elif p_value < 0.01:
        return f"p = {p_value:.3f}**"
    elif p_value < 0.05:
        return f"p = {p_value:.3f}*"
    else:
        return f"p = {p_value:.3f}"

def generate_plots(df_raw, stats_results_df):
    """
    Generates and saves violin plots with jitter for all scenarios,
    comparing 'fit' and 'rfit' run types, with p-value annotations.
    """
    if df_raw.empty:
        print("Warning: Raw data is empty. Skipping plot generation.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    sns.set_theme(style="whitegrid") 
    run_type_palette = {'fit': 'skyblue', 'rfit': 'lightcoral'}
    metrics_to_plot = ['TimeSec', 'RSSDeltaKB']

    all_scenarios = df_raw['ScenarioName'].unique()
    
    for scenario in all_scenarios:
        scenario_df = df_raw[df_raw['ScenarioName'] == scenario]
        if scenario_df.empty:
            print(f"Warning: No data for scenario {scenario}. Skipping plots.")
            continue
        
        backend_order = sorted(scenario_df['BackendName'].unique())

        for metric in metrics_to_plot:
            if metric not in scenario_df.columns or scenario_df[metric].isnull().all():
                print(f"Warning: Metric {metric} not found or all NaN for scenario {scenario}. Skipping plot for this metric.")
                continue

            fig, ax = plt.subplots(figsize=(max(10, len(backend_order) * 1.8), 8))
            
            sns.violinplot(x='BackendName', y=metric, hue='RunType', data=scenario_df,
                           palette=run_type_palette, dodge=True, order=backend_order,
                           inner=None, ax=ax)
            
            sns.stripplot(x='BackendName', y=metric, hue='RunType', data=scenario_df,
                          size=3, color='gray', dodge=True, jitter=True, alpha=0.6, 
                          order=backend_order, ax=ax)
            
            plot_title_metric = metric.replace("Sec", " (s)").replace("DeltaKB", " (KB)")
            ax.set_title(f'{plot_title_metric}: {scenario} (Fit vs RFit)', fontsize=16)
            ax.set_ylabel(plot_title_metric, fontsize=12)
            ax.set_xlabel('Backend', fontsize=12)
            ax.tick_params(axis='x', labelsize=10, rotation=30, ha="right")
            ax.tick_params(axis='y', labelsize=10)
            
            current_handles, current_labels = ax.get_legend_handles_labels()
            num_hue_levels = len(scenario_df['RunType'].unique())
            if len(current_handles) > num_hue_levels: # Avoid duplicate legend items from stripplot
                 ax.legend(current_handles[:num_hue_levels], current_labels[:num_hue_levels], title='Run Type', loc='upper right')
            else:
                 ax.legend(title='Run Type', loc='upper right')


            y_min_data, y_max_data = scenario_df[metric].min(), scenario_df[metric].max()
            if pd.isna(y_min_data) or pd.isna(y_max_data): # If metric data is all NaN
                y_min_data, y_max_data = ax.get_ylim() # Fallback to current plot limits

            plot_y_range = y_max_data - y_min_data
            if plot_y_range == 0 or pd.isna(plot_y_range): plot_y_range = abs(y_max_data) if pd.notna(y_max_data) else 1.0


            annotation_y_offset_initial = plot_y_range * 0.05 
            annotation_y_step = plot_y_range * 0.03 
            current_max_y_for_annotations = y_max_data 

            for i, backend_name in enumerate(backend_order):
                test_result_row = stats_results_df[
                    (stats_results_df['ScenarioName'] == scenario) &
                    (stats_results_df['BackendName'] == backend_name) &
                    (stats_results_df['Metric'] == metric) &
                    (stats_results_df['TestType'] == 'Mann-Whitney U (fit vs rfit)')
                ]

                if not test_result_row.empty and 'P-Value' in test_result_row.columns:
                    p_value = test_result_row['P-Value'].iloc[0]
                    error_msg = test_result_row['Error'].iloc[0] if 'Error' in test_result_row.columns and pd.notna(test_result_row['Error'].iloc[0]) else None

                    if error_msg:
                        # print(f"Skipping annotation for {scenario}/{backend_name}/{metric} due to test error: {error_msg}")
                        annotation_text = "Test Error"
                    elif pd.notna(p_value):
                        annotation_text = format_p_value(p_value)
                    else:
                        # print(f"Skipping annotation for {scenario}/{backend_name}/{metric} (p-value is NaN, no error reported)")
                        annotation_text = "n/a (no test)" # p-value is NaN, no specific error

                    # Determine y for annotation elements
                    backend_metric_data = scenario_df[(scenario_df['BackendName'] == backend_name)][metric]
                    y_max_for_backend = backend_metric_data.max() if not backend_metric_data.empty else y_max_data
                    if pd.isna(y_max_for_backend): y_max_for_backend = y_max_data

                    line_y = max(y_max_for_backend + annotation_y_offset_initial, current_max_y_for_annotations + annotation_y_step)
                    text_y = line_y + annotation_y_step * 0.5 
                    
                    x1 = i - 0.2  # Approximate x for 'fit' (left violin of the pair)
                    x2 = i + 0.2  # Approximate x for 'rfit' (right violin of the pair)
                    
                    cap_height = plot_y_range * 0.01
                    ax.plot([x1, x1, x2, x2], [line_y - cap_height, line_y, line_y, line_y - cap_height], lw=1.0, color='dimgray')
                    ax.text((x1 + x2) / 2, text_y, annotation_text, ha='center', va='bottom', fontsize=8, color='dimgray') # Smaller font
                    
                    current_max_y_for_annotations = text_y # Update for next annotation to be placed above this one

            # Adjust overall plot limits after all annotations
            final_y_min, final_y_max = ax.get_ylim()
            ax.set_ylim(final_y_min, max(final_y_max, current_max_y_for_annotations + annotation_y_offset_initial))
            
            plt.tight_layout()
            
            sanitized_scenario = scenario.replace(" ", "_").replace("/", "-")
            plot_filename = os.path.join(OUTPUT_DIR, f'plot_{metric.lower()}_{sanitized_scenario}.png')
            try:
                plt.savefig(plot_filename, dpi=150)
                print(f"Saved plot: {plot_filename}")
            except Exception as e:
                print(f"Error saving plot {plot_filename}: {e}")
            plt.close(fig)

def perform_statistical_tests(df_raw):
    """
    Performs Kruskal-Wallis tests for overall backend comparison (within a run_type for key scenarios)
    and pairwise Mann-Whitney U tests comparing 'fit' vs 'rfit' for each backend (for all scenarios).
    """
    if df_raw.empty:
        print("Warning: Raw data is empty. Skipping statistical tests.")
        return pd.DataFrame()
        
    results_list = []
    metrics_to_test = ['TimeSec', 'RSSDeltaKB']
    
    all_scenarios = df_raw['ScenarioName'].unique()
    all_backends = df_raw['BackendName'].unique()
    all_runtypes = df_raw['RunType'].unique()

    # Kruskal-Wallis tests (Overall comparison of backends for a given RunType and KEY SCENARIO)
    for scenario in KEY_SCENARIOS_FOR_PLOTS_STATS: 
        if scenario not in all_scenarios: 
            # print(f"K-W: Skipping key scenario '{scenario}' not found in data.")
            continue
        for run_type in all_runtypes:
            for metric in metrics_to_test:
                kw_df = df_raw[
                    (df_raw['ScenarioName'] == scenario) & 
                    (df_raw['RunType'] == run_type) &
                    df_raw[metric].notna()
                ]
                if kw_df.empty: continue

                unique_backends_in_kw_df = kw_df['BackendName'].unique()
                if len(unique_backends_in_kw_df) < 2: continue

                grouped_data_kw = [kw_df[kw_df['BackendName'] == backend][metric] for backend in unique_backends_in_kw_df]
                if any(len(group) < 2 for group in grouped_data_kw): continue

                try:
                    h_stat, p_value = stats.kruskal(*grouped_data_kw)
                    results_list.append({
                        'ScenarioName': scenario, 'RunType': run_type, 'BackendName': 'Overall', 'Metric': metric,
                        'TestType': 'Kruskal-Wallis (backends)', 'H-Statistic': h_stat, 'P-Value': p_value, 'Error':None
                    })
                except ValueError as e:
                    results_list.append({
                        'ScenarioName': scenario, 'RunType': run_type, 'BackendName': 'Overall', 'Metric': metric,
                        'TestType': 'Kruskal-Wallis (backends)', 'H-Statistic': np.nan, 'P-Value': np.nan, 'Error': str(e)
                    })

    # Pairwise Mann-Whitney U tests (fit vs rfit for each Backend and ALL Scenarios)
    for scenario in all_scenarios: 
        for backend_name in all_backends:
            for metric in metrics_to_test:
                base_record = {'ScenarioName': scenario, 'BackendName': backend_name, 'Metric': metric, 
                               'TestType': 'Mann-Whitney U (fit vs rfit)', 'RunTypeComparison': 'fit_vs_rfit'}
                fit_data = df_raw[(df_raw['ScenarioName'] == scenario) & (df_raw['BackendName'] == backend_name) & (df_raw['RunType'] == 'fit') & df_raw[metric].notna()][metric]
                rfit_data = df_raw[(df_raw['ScenarioName'] == scenario) & (df_raw['BackendName'] == backend_name) & (df_raw['RunType'] == 'rfit') & df_raw[metric].notna()][metric]

                if fit_data.empty or rfit_data.empty:
                    results_list.append({**base_record, 'U-Statistic': np.nan, 'P-Value': np.nan, 'Error': 'Insufficient data for one or both groups (fit/rfit)'})
                    continue
                
                # Check for zero variance data if scipy version is old, otherwise mannwhitneyu might handle it.
                if len(np.unique(fit_data)) == 1 and len(np.unique(rfit_data)) == 1 and np.unique(fit_data)[0] == np.unique(rfit_data)[0]:
                    # Both groups have same constant value, p should be 1 (no difference) or test might error.
                    # Scipy > 1.6.0 handles this with p=1. Older versions might error.
                    # For robustness, we can assign p=1 in this specific edge case.
                    u_stat, p_value = 0, 1.0 # Or specific U-statistic if known for this case
                    error_msg = "Both groups constant and equal" 
                    # This is not really an "error" for the test, but a specific condition.
                    # Depending on scipy version, this might not be needed.
                    # For now, let mannwhitneyu handle it and report error if it occurs.
                
                try:
                    u_stat, p_value = stats.mannwhitneyu(fit_data, rfit_data, alternative='two-sided', use_continuity=True)
                    results_list.append({**base_record, 'U-Statistic': u_stat, 'P-Value': p_value, 'Error': None})
                except ValueError as e: 
                    results_list.append({**base_record, 'U-Statistic': np.nan, 'P-Value': np.nan, 'Error': str(e)})
    
    if not results_list: # Ensure a DataFrame with correct columns is returned even if empty
        return pd.DataFrame(columns=['ScenarioName', 'RunType', 'BackendName', 'Metric', 'TestType', 
                                     'H-Statistic', 'P-Value', 'Error', 'U-Statistic', 'RunTypeComparison'])
    return pd.DataFrame(results_list)


def generate_performance_vs_factor_plots(df_raw, stats_results_df):
    """
    Generates plots of performance metrics vs. varying factors (NumSamples or NumFeatures)
    for each scenario.
    """
    if df_raw.empty:
        print("Warning: Raw data is empty. Skipping performance vs. factor plot generation.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    sns.set_theme(style="whitegrid")
    metrics_to_plot = ['TimeSec', 'RSSDeltaKB']
    factors_to_check = ['NumSamples', 'NumFeatures']
    
    all_scenarios = df_raw['ScenarioName'].unique()

    for scenario in all_scenarios:
        scenario_df = df_raw[df_raw['ScenarioName'] == scenario].copy() # Use .copy() to avoid SettingWithCopyWarning

        if scenario_df.empty:
            # print(f"Info: No data for scenario '{scenario}' in performance vs. factor plots.")
            continue

        for factor in factors_to_check:
            if factor not in scenario_df.columns:
                # print(f"Info: Factor '{factor}' not found in scenario '{scenario}'.")
                continue
            
            unique_factor_values = scenario_df[factor].nunique()
            if unique_factor_values <= 1:
                # This factor does not vary in this scenario, so skip plotting against it.
                # print(f"Info: Factor '{factor}' does not vary for scenario '{scenario}'. Skipping.")
                continue

            # Ensure factor is numeric for plotting on x-axis
            scenario_df[factor] = pd.to_numeric(scenario_df[factor], errors='coerce')
            scenario_df.dropna(subset=[factor], inplace=True) # Remove rows where factor could not be coerced

            if scenario_df[factor].nunique() <= 1: # Check again after coercion
                # print(f"Info: Factor '{factor}' has <= 1 unique numeric value for scenario '{scenario}' after coercion. Skipping.")
                continue

            for metric in metrics_to_plot:
                if metric not in scenario_df.columns or scenario_df[metric].isnull().all():
                    print(f"Warning: Metric '{metric}' not found or all NaN for scenario '{scenario}' (factor: {factor}). Skipping plot.")
                    continue
                
                # Ensure metric is numeric
                scenario_df[metric] = pd.to_numeric(scenario_df[metric], errors='coerce')
                # It's important to handle NaNs in the metric column, lineplot might struggle or exclude them.
                # We can drop rows where the current metric is NaN, but this should be done carefully
                # if other metrics for the same row are valid. For lineplot, it's usually fine.
                plot_df = scenario_df.dropna(subset=[metric])

                if plot_df.empty:
                    print(f"Warning: No data for metric '{metric}' after NaN removal in scenario '{scenario}' (factor: {factor}).")
                    continue


                plt.figure(figsize=(12, 7))
                try:
                    ax = sns.lineplot(data=plot_df, x=factor, y=metric, hue='BackendName', style='RunType', markers=True, errorbar=('ci', 95))
                except Exception as e:
                    print(f"Error during lineplot for {metric} vs {factor} in {scenario}: {e}")
                    plt.close() # Close the figure if plot fails
                    continue


                plot_title_metric = metric.replace("Sec", " (s)").replace("DeltaKB", " (KB)")
                factor_display_name = factor.replace("Num", "Number of ")
                
                ax.set_title(f'{plot_title_metric} vs. {factor_display_name} for {scenario}', fontsize=16)
                ax.set_xlabel(factor_display_name, fontsize=12)
                ax.set_ylabel(plot_title_metric, fontsize=12)
                
                # Improve x-axis ticks if factor is NumSamples or NumFeatures - show all unique values if reasonable
                unique_x_values = sorted(plot_df[factor].unique())
                if len(unique_x_values) <= 10: # Show all ticks if 10 or fewer unique x values
                     ax.set_xticks(unique_x_values)
                
                plt.xticks(rotation=30, ha="right")
                plt.yticks(fontsize=10)
                plt.legend(title='Backend / RunType', loc='upper left', bbox_to_anchor=(1, 1)) # Move legend outside
                
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside

                sanitized_scenario = scenario.replace(" ", "_").replace("/", "-")
                sanitized_factor = factor.replace(" ", "_")
                plot_filename = os.path.join(OUTPUT_DIR, f'plot_line_{metric.lower()}_vs_{sanitized_factor}_{sanitized_scenario}.png')
                
                try:
                    plt.savefig(plot_filename, dpi=150, bbox_inches='tight') # Use bbox_inches for external legend
                    print(f"Saved plot: {plot_filename}")
                except Exception as e:
                    print(f"Error saving plot {plot_filename}: {e}")
                plt.close()


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR); print(f"Created output directory: {OUTPUT_DIR}")
    
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR); print(f"Created artifacts directory (for local testing): {ARTIFACTS_DIR}")

    raw_df = load_data(ARTIFACTS_DIR)
    if raw_df.empty:
        print(f"No data loaded from {ARTIFACTS_DIR}. Attempting to use dummy data for demonstration.")
        # Dummy data for testing the plotting and stats logic if no files are present
        sample_data = {
            'ScenarioName': ['TestScenario1', 'TestScenario1', 'TestScenario1', 'TestScenario1', 'TestScenario1', 'TestScenario1', 'TestScenario2', 'TestScenario2'],
            'BackendName': ['BackendA', 'BackendA', 'BackendB', 'BackendB', 'BackendC', 'BackendC', 'BackendA', 'BackendA'],
            'RunType': ['fit', 'rfit', 'fit', 'rfit', 'fit', 'rfit', 'fit', 'rfit'],
            'TimeSec': [1.0, 0.5, 1.2, 0.6, 0.8, 0.85, 2.0, 1.5], # BackendC rfit is slightly worse
            'RSSDeltaKB': [100,50,120,60, 90, 95, 200, 150],
            'NumSamples':[100]*8, 'NumFeatures':[10]*8, 'Iteration': list(range(1,9)) # Dummy iteration
        }
        # Add more samples for each group to make stats more meaningful
        expanded_sample_data = {key: [] for key in sample_data.keys()}
        for i in range(len(sample_data['ScenarioName'])):
            for _ in range(5): # 5 samples per original entry
                for key in sample_data.keys():
                    expanded_sample_data[key].append(sample_data[key][i] + (np.random.rand()*0.1-0.05 if key in ['TimeSec', 'RSSDeltaKB'] else 0) )


        raw_df = pd.DataFrame(expanded_sample_data)
        if raw_df.empty: # If dummy data creation also fails for some reason
             print("Dummy data creation failed. Exiting analysis script.")
             return
        print("Loaded dummy data for testing purposes.")


    raw_df = clean_data(raw_df)
    if raw_df.empty: 
        print("Data became empty after cleaning. Exiting.")
        return

    aggregated_df = aggregate_data(raw_df.copy()) 

    if not aggregated_df.empty:
        agg_file_path = os.path.join(OUTPUT_DIR, "consolidated_benchmark_analysis.tsv")
        try:
            aggregated_df.to_csv(agg_file_path, sep='\t', index=False, float_format='%.6f')
            print(f"Consolidated analysis TSV saved to: {agg_file_path}")
        except Exception as e:
            print(f"Error saving consolidated analysis: {e}")
    else:
        print("Aggregated data is empty. Skipping saving consolidated_benchmark_analysis.tsv.")

    stats_results_df = perform_statistical_tests(raw_df)
    generate_plots(raw_df, stats_results_df) 
    generate_performance_vs_factor_plots(raw_df, stats_results_df) # Call the new function

    print("\nStatistical Test Summary:")
    if stats_results_df is not None and not stats_results_df.empty:
        # Optionally filter or sort before printing/saving for clarity
        # For example, to see only Mann-Whitney results for plots:
        # mw_results = stats_results_df[stats_results_df['TestType'] == 'Mann-Whitney U (fit vs rfit)']
        # print(mw_results.to_string())
        print(stats_results_df.to_string())
        stats_file_path = os.path.join(OUTPUT_DIR, "statistical_analysis_summary.tsv")
        try:
            stats_results_df.to_csv(stats_file_path, sep='\t', index=False, float_format='%.6f')
            print(f"Statistical analysis summary saved to: {stats_file_path}")
        except Exception as e:
            print(f"Error saving statistical analysis summary: {e}")
    else:
        print("No statistical tests performed or no results generated.")

if __name__ == "__main__":
    main()
