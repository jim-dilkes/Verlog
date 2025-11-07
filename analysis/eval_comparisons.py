import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from analysis_configs import legend_kwargs, run_name_mapping, run_env_mapping, run_alg_mapping, run_name_colors, eval_env_name_mapping
from run_eval_metrics_clean import run_eval_metrics_clean

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# Set Seaborn style for better looking plots
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")  # More accessible color palette
# Import configurations from default_configs.py
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',  
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'figure.figsize': (10, 6),  # Good size for single-column papers
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'grid.alpha': 0.3
})


# 1000 samples data
dqn_data = {
    'Environment': [
        'FrozenLake-NoSlip',
        'FrozenLake-Slippery',
        'Snake-20Step', 
        'Snake-20Steps-Banana',
        'Snake-20Steps-PoisonApple',
        'Snake-20Steps-DownHill',
    ],
    'DQN Mean': [
        0.1700,
        0.1750,
        4.5768,
        11.8619,
        -5.6312,
        5.0628
    ],
    'DQN Std': [
        0.3758,
        0.3802,
        2.4692,
        10.3822,
        2.2520,
        2.9668
    ],
    'DQN CI 95': [
        0.104,
        0.105,
        0.684,
        2.878,
        0.624,
        0.822
    ]
}
dqn_df = pd.DataFrame(dqn_data)

# DQN environment mapping to target environments
# Only include environments where DQN data is available and matches target environments
dqn_env_mapping = {
    'FrozenLake-NoSlip': 'FrozenLake-NoSlip',
    'FrozenLake-Slippery': 'FrozenLake-Slippery',
    'Snake-20Step': 'Snake-20Step'
}

# ------------------
# Statistical Test Functions
# ------------------

def perform_anova_test(df, metric_col):
    """
    Perform two-way ANOVA with interaction to test if factors significantly influence 
    the degree of improvement (Delta = Score_Step550 - Score_Step0).
    
    Args:
        df: DataFrame with columns for Algorithm, Model_Size, Environment, Score_Step0, Score_Step550
        metric_col: The metric column name (e.g., 'eval_FastSnake-Default/score_mean')
    
    Returns:
        dict: ANOVA results including p-values for main effects and interactions
    """
    try:
        # Create a copy of the data for this analysis
        anova_data = df.copy()
        
        # Extract factors from group names
        anova_data['Algorithm'] = anova_data['group'].str.extract(r'_(PPO|GRPO)_')[0]
        anova_data['Model_Size'] = anova_data['group'].str.extract(r'_(0pt5B|3B|7B)')[0]
        anova_data['Task'] = anova_data['group'].str.extract(r'^(FL|FS)_')[0]
        
        # Check for na's, print any:
        print(f"NA's in Algorithm: {anova_data['Algorithm'].isna().sum()}")
        print(f"NA's in Model_Size: {anova_data['Model_Size'].isna().sum()}")
        print(f"NA's in Task: {anova_data['Task'].isna().sum()}")
        
        # Remove rows where we couldn't extract the factors
        anova_data = anova_data.dropna(subset=['Algorithm', 'Model_Size', 'Task'])
        
        # Map model sizes to more readable format
        anova_data['Model_Size'] = anova_data['Model_Size'].map({
            '0pt5B': '0.5B',
            '3B': '3B', 
            '7B': '7B',
            '14B': '14B'
        })
        
        # Map task names
        anova_data['Task'] = anova_data['Task'].map({
            'FL': 'FrozenLake',
            'FS': 'Snake'
        })
        
        # Extract environment name from metric column
        env_name = metric_col.replace('eval_', '').replace('/score_mean', '').replace('/rewards_mean', '')
        anova_data['Environment'] = env_name
        
        # Get step 0 and 550 data
        step_0_data = anova_data[anova_data['_step'] == 0]
        step_550_data = anova_data[anova_data['_step'] == 550]
        
        # Merge step 0 and 550 data to calculate Delta
        merged_data = step_0_data.merge(
            step_550_data, 
            on=['group', 'experiment_name'], 
            suffixes=('_step0', '_step550')
        )
        
        if len(merged_data) == 0:
            return {'error': 'No paired data found for ANOVA'}
        
        print(f"  Merged data shape: {merged_data.shape}")
        print(f"  Available columns: {list(merged_data.columns)}")
        
        # Calculate Delta (improvement)
        score_step0_col = f'{metric_col}_step0'
        score_step550_col = f'{metric_col}_step550'
        
        if score_step0_col not in merged_data.columns or score_step550_col not in merged_data.columns:
            return {'error': f'Required columns not found: {score_step0_col}, {score_step550_col}'}
        
        merged_data['Delta'] = merged_data[score_step550_col] - merged_data[score_step0_col]
        
        # Remove rows with NaN values
        merged_data = merged_data.dropna(subset=['Delta', 'Algorithm', 'Model_Size', 'Task'])
        
        print(f"  After removing NaN: {len(merged_data)} rows")
        if len(merged_data) > 0:
            print(f"  Algorithm values: {merged_data['Algorithm'].unique()}")
            print(f"  Model_Size values: {merged_data['Model_Size'].unique()}")
            print(f"  Task values: {merged_data['Task'].unique()}")
        
        if len(merged_data) < 2:
            return {'error': 'Insufficient data for ANOVA'}
        
        try:
            # Perform two-way ANOVA with interaction
            # Using Type 2 Sum of Squares as recommended when interaction terms are present
            formula = 'Delta ~ C(Algorithm) * C(Model_Size) * C(Task)'
            print(f"  Formula: {formula}")
            print(f"  Data shape: {merged_data.shape}")
            print(f"  Delta values: {merged_data['Delta'].describe()}")
            
            model = ols(formula, data=merged_data).fit()
            anova_table = anova_lm(model, typ=2)
            
            print(f"  ANOVA table index: {anova_table.index.tolist()}")
            
            # Extract p-values
            results = {
                'Algorithm_p': anova_table.loc['C(Algorithm)', 'PR(>F)'] if 'C(Algorithm)' in anova_table.index else np.nan,
                'Model_Size_p': anova_table.loc['C(Model_Size)', 'PR(>F)'] if 'C(Model_Size)' in anova_table.index else np.nan,
                'Task_p': anova_table.loc['C(Task)', 'PR(>F)'] if 'C(Task)' in anova_table.index else np.nan,
                'Algorithm_Model_Size_p': anova_table.loc['C(Algorithm):C(Model_Size)', 'PR(>F)'] if 'C(Algorithm):C(Model_Size)' in anova_table.index else np.nan,
                'Algorithm_Task_p': anova_table.loc['C(Algorithm):C(Task)', 'PR(>F)'] if 'C(Algorithm):C(Task)' in anova_table.index else np.nan,
                'Model_Size_Task_p': anova_table.loc['C(Model_Size):C(Task)', 'PR(>F)'] if 'C(Model_Size):C(Task)' in anova_table.index else np.nan,
                'Algorithm_Model_Size_Task_p': anova_table.loc['C(Algorithm):C(Model_Size):C(Task)', 'PR(>F)'] if 'C(Algorithm):C(Model_Size):C(Task)' in anova_table.index else np.nan,
                'n_observations': len(merged_data),
                'anova_table': anova_table
            }
        except Exception as e:
            print(f"  ANOVA error: {e}")
            print(f"  Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return {'error': f'ANOVA failed: {e}'}
        
        return results
        
    except Exception as e:
        return {'error': f'ANOVA failed: {str(e)}'}

def perform_levenes_test(df, metric_col):
    """
    Perform Levene's test to check if training process significantly changed 
    variance (stability) across factor groupings.
    
    Args:
        df: DataFrame with evaluation data
        metric_col: The metric column name (e.g., 'eval_FastSnake-Default/score_mean')
    
    Returns:
        dict: Levene's test results for each configuration
    """
    try:
        # Get step 0 and 550 data
        step_0_data = df[df['_step'] == 0][metric_col].dropna()
        step_550_data = df[df['_step'] == 550][metric_col].dropna()
        
        if len(step_0_data) < 2 or len(step_550_data) < 2:
            return {'error': 'Insufficient data for Levene test'}
        
        # Perform Levene's test
        statistic, p_value = stats.levene(step_0_data, step_550_data)
        
        # Determine significance
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        return {
            'levene_statistic': statistic,
            'levene_p_value': p_value,
            'levene_significance': significance,
            'step_0_variance': step_0_data.var(),
            'step_550_variance': step_550_data.var(),
            'n_step_0': len(step_0_data),
            'n_step_550': len(step_550_data)
        }
        
    except Exception as e:
        return {'error': f'Levene test failed: {str(e)}'}

# ------------------
# Load the data
# ------------------

# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(current_dir, 'analysis/data')
figures_dir = os.path.join(current_dir, 'analysis/charts')
os.makedirs(figures_dir, exist_ok=True) 

run_eval_metrics_clean = run_eval_metrics_clean(data_dir)

print(f"Distinct groups: {run_eval_metrics_clean['group'].unique()}")
print(f"Distinct experiment names: {run_eval_metrics_clean['experiment_name'].unique()}")
print(f"Distinct group run numbers: {run_eval_metrics_clean['group_run_number'].unique()}")

# 'FL_GRPO_0pt5B_run5_slurm340889'
# 'FS_GRPO_0pt5B_run4_slurm340887'
# 'FL_GRPO_0pt5B_run5_slurm340884'

allowed_group_run_numbers = [
#  'FL_GRPO_0pt5B_run1tkzv81f',
#  'FL_GRPO_0pt5B_runatl66ff2',
#  'FL_GRPO_0pt5B_runk53h1wsr',
#  'FL_GRPO_0pt5B_runr4s6i5s0',
#  'FL_PPO_0pt5B_runpr4rtyyk',
#  'FL_PPO_0pt5B_runyvi30du4',
#  'FL_PPO_0pt5B_runzkiv53u6',
#  'FS_GRPO_0pt5B_run2q1jongf',
#  'FS_GRPO_0pt5B_run6k4ys2mn',
#  'FS_GRPO_0pt5B_runzpqb8hrr',
#  'FS_PPO_0pt5B_runfzuay1sj',
#  'FS_PPO_0pt5B_runhzkacmiv',
#  'FS_PPO_0pt5B_runih18s3xc',
#  'FS_PPO_0pt5B_runya8jsldm',
 'FL_GRPO_3B_run10',
 'FL_GRPO_3B_run11' ,
 'FL_GRPO_3B_run16',
 'FL_GRPO_3B_run17',
 'FL_GRPO_3B_run18' ,
 'FL_PPO_3B_run10' ,
 'FL_PPO_3B_run11',
 'FL_PPO_3B_run13',
 'FL_PPO_3B_run2' ,
 'FL_PPO_3B_run1' ,
 'FS_GRPO_3B_run16' ,
 'FS_GRPO_3B_run18',
 'FS_GRPO_3B_run20',
 'FS_GRPO_3B_run21' ,
 'FS_GRPO_3B_run22',
#  'FS_GRPO_3B_run30',
#  'FS_GRPO_3B_run31' ,
#  'FS_GRPO_3B_run32',
#  'FS_GRPO_3B_run4',
#  'FS_GRPO_3B_run5' ,
 'FS_PPO_3B_run10',
 'FS_PPO_3B_run11',
 'FS_PPO_3B_run12' ,
 'FS_PPO_3B_run13' ,
#  'FS_PPO_3B_run3',
 'FS_PPO_7B_run40',
 'FS_PPO_7B_run41',
 'FS_PPO_7B_run42',
 'FS_PPO_7B_run43',
 'FS_PPO_7B_run44',
 'FS_PPO_7B_run30',
 'FS_PPO_7B_run31',
 'FS_PPO_14B_run1',
 'FS_PPO_14B_run2',
 'FS_PPO_14B_run5',
 ]

run_eval_metrics_clean = run_eval_metrics_clean[run_eval_metrics_clean['group_run_number'].isin(allowed_group_run_numbers)]
print(f"Filtered run_eval_metrics_clean: {len(run_eval_metrics_clean)}")

# ------------------
# Create Before/After Comparison Table
# ------------------

def create_before_after_table(use_score=False):
    """
    Create a table comparing evaluation scores at:
    1. Step 0 vs Step 550 (initial vs final)
    2. Step 50 vs Step 550 (early vs final, to distinguish formatting vs learning)
    for specific metrics across all groups, including statistical significance testing.
    """
    print("\n" + "="*60)
    print("CREATING BEFORE/AFTER COMPARISON TABLE WITH STATISTICAL TESTS")
    print("="*60)
    print("Comparing: Step 0 vs 550 (initial vs final) and Step 50 vs 550 (early vs final)")
    
    # Define the metrics we want to compare (removed val/ and FrozenLake-BigSlippery)
    target_metrics = [
        'eval_FastSnake-Default/score_mean',
        'eval_Snake-20Step/score_mean', 
        'eval_FrozenLake-NoSlip/score_mean',
        'eval_FrozenLake-Slippery/score_mean'
    ] if use_score else [
        'eval_FastSnake-Default/rewards_mean',
        'eval_Snake-20Step/rewards_mean', 
        'eval_FrozenLake-NoSlip/rewards_mean',
        'eval_FrozenLake-Slippery/rewards_mean'
    ]
    
    # Check which metrics are available in the data
    available_metrics = [col for col in run_eval_metrics_clean.columns if col.startswith('eval_') or col.startswith('val/')]
    print(f"Available metrics: {len(available_metrics)}")
    
    # Filter to only the metrics we want that are actually available
    metrics_to_use = [metric for metric in target_metrics if metric in available_metrics]
    print(f"Metrics to analyze: {metrics_to_use}")
    
    # Get unique groups
    groups = run_eval_metrics_clean['group'].unique()
    print(f"Groups found: {groups}")
    
    # Create the comparison table
    comparison_data = []
    statistical_results = []
    anova_results = []
    levene_results = []
    
    def perform_comparison(group_data, metric, step_before, step_after, comparison_name):
        """Helper function to perform a single comparison between two steps."""
        # Get data for both steps
        step_before_data = group_data[group_data['_step'] == step_before][metric].dropna()
        step_after_data = group_data[group_data['_step'] == step_after][metric].dropna()
        
        # Calculate means and stds
        step_before_mean = step_before_data.mean() if len(step_before_data) > 0 else np.nan
        step_before_std = step_before_data.std() if len(step_before_data) > 0 else np.nan
        step_after_mean = step_after_data.mean() if len(step_after_data) > 0 else np.nan
        step_after_std = step_after_data.std() if len(step_after_data) > 0 else np.nan
        
        # Calculate improvement
        improvement = step_after_mean - step_before_mean if not (np.isnan(step_after_mean) or np.isnan(step_before_mean)) else np.nan
        
        # Perform paired t-test for statistical significance
        p_value = np.nan
        t_stat = np.nan
        significant = "N/A"
        n_pairs = 0
        
        if len(step_before_data) > 0 and len(step_after_data) > 0:
            # For paired t-test, we need to match runs using group_run_number
            # (since experiment_name includes slurm IDs that change between checkpoint restarts)
            step_before_runs = group_data[group_data['_step'] == step_before]['group_run_number'].tolist()
            step_after_runs = group_data[group_data['_step'] == step_after]['group_run_number'].tolist()
            
            # Find common runs
            common_runs = set(step_before_runs) & set(step_after_runs)
            
            if len(common_runs) >= 2:  # Need at least 2 pairs for t-test
                # Get paired data
                paired_before = []
                paired_after = []
                
                for run in common_runs:
                    before_val = group_data[(group_data['_step'] == step_before) & (group_data['group_run_number'] == run)][metric].iloc[0]
                    after_val = group_data[(group_data['_step'] == step_after) & (group_data['group_run_number'] == run)][metric].iloc[0]
                    
                    if not (np.isnan(before_val) or np.isnan(after_val)):
                        paired_before.append(before_val)
                        paired_after.append(after_val)
                
                n_pairs = len(paired_before)
                if n_pairs >= 2:
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_rel(paired_after, paired_before)
                    
                    # Determine significance
                    if p_value < 0.001:
                        significant = "***"
                    elif p_value < 0.01:
                        significant = "**"
                    elif p_value < 0.05:
                        significant = "*"
                    else:
                        significant = "ns"
        
        return {
            'before_mean': step_before_mean,
            'before_std': step_before_std,
            'after_mean': step_after_mean,
            'after_std': step_after_std,
            'improvement': improvement,
            'p_value': p_value,
            't_stat': t_stat,
            'significant': significant,
            'n_pairs': n_pairs
        }
    
    for group in groups:
        group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
        
        # Get mapped group name
        mapped_group_name = run_name_mapping.get(group, group)
        
        row_data = {'Group': mapped_group_name}
        
        for metric in metrics_to_use:
            # Format the metric name for display
            if metric.startswith('val/'):
                display_name = 'Validation Reward'
            else:
                # Extract environment name and apply mapping
                env_name_raw = metric.replace('eval_', '').replace('/rewards_mean', '').replace('/score_mean', '')
                display_name = eval_env_name_mapping.get(env_name_raw, env_name_raw)
            
            # Perform both comparisons: 0 vs 550 and 50 vs 550
            comparison_0_550 = perform_comparison(group_data, metric, 0, 550, "Initial vs Final")
            comparison_50_550 = perform_comparison(group_data, metric, 50, 550, "Early vs Final")
            
            # Store the data for 0 vs 550 comparison (all models)
            row_data[f'{display_name} (Step 0)'] = f"{comparison_0_550['before_mean']:.3f} ± {comparison_0_550['before_std']:.3f}" if not np.isnan(comparison_0_550['before_mean']) else "N/A"
            row_data[f'{display_name} (Step 550)'] = f"{comparison_0_550['after_mean']:.3f} ± {comparison_0_550['after_std']:.3f}" if not np.isnan(comparison_0_550['after_mean']) else "N/A"
            row_data[f'{display_name} (0→550)'] = f"{comparison_0_550['improvement']:+.3f}" if not np.isnan(comparison_0_550['improvement']) else "N/A"
            row_data[f'{display_name} (0→550 p)'] = f"{comparison_0_550['p_value']:.4f}" if not np.isnan(comparison_0_550['p_value']) else "N/A"
            row_data[f'{display_name} (0→550 sig)'] = comparison_0_550['significant']
            
            # Check if this is a 0.5B model
            is_0pt5b_model = '0pt5B' in group
            
            # Store the data for 50 vs 550 comparison (only for 0.5B models)
            if is_0pt5b_model:
                row_data[f'{display_name} (Step 50)'] = f"{comparison_50_550['before_mean']:.3f} ± {comparison_50_550['before_std']:.3f}" if not np.isnan(comparison_50_550['before_mean']) else "N/A"
                row_data[f'{display_name} (50→550)'] = f"{comparison_50_550['improvement']:+.3f}" if not np.isnan(comparison_50_550['improvement']) else "N/A"
                row_data[f'{display_name} (50→550 p)'] = f"{comparison_50_550['p_value']:.4f}" if not np.isnan(comparison_50_550['p_value']) else "N/A"
                row_data[f'{display_name} (50→550 sig)'] = comparison_50_550['significant']
            else:
                # For non-0.5B models, show N/A for 50→550 columns
                row_data[f'{display_name} (Step 50)'] = "N/A"
                row_data[f'{display_name} (50→550)'] = "N/A"
                row_data[f'{display_name} (50→550 p)'] = "N/A"
                row_data[f'{display_name} (50→550 sig)'] = "N/A"
            
            # Store statistical results for summary
            if not np.isnan(comparison_0_550['p_value']):
                statistical_results.append({
                    'Group': mapped_group_name,
                    'Metric': display_name,
                    'Comparison': '0→550',
                    'Improvement': comparison_0_550['improvement'],
                    'p_value': comparison_0_550['p_value'],
                    't_statistic': comparison_0_550['t_stat'],
                    'Significant': comparison_0_550['significant'],
                    'n_pairs': comparison_0_550['n_pairs']
                })
            
            # Only store 50→550 results for 0.5B models
            if is_0pt5b_model and not np.isnan(comparison_50_550['p_value']):
                statistical_results.append({
                    'Group': mapped_group_name,
                    'Metric': display_name,
                    'Comparison': '50→550',
                    'Improvement': comparison_50_550['improvement'],
                    'p_value': comparison_50_550['p_value'],
                    't_statistic': comparison_50_550['t_stat'],
                    'Significant': comparison_50_550['significant'],
                    'n_pairs': comparison_50_550['n_pairs']
                })
        
        comparison_data.append(row_data)
    
    # Perform ANOVA and Levene tests for each metric
    print("\nPerforming ANOVA and Levene tests...")
    for metric in metrics_to_use:
        print(f"  Testing metric: {metric}")
        
        # Perform ANOVA test
        anova_result = perform_anova_test(run_eval_metrics_clean, metric)
        if 'error' not in anova_result:
            anova_result['metric'] = metric
            anova_results.append(anova_result)
            print(f"    ANOVA completed - n_observations: {anova_result['n_observations']}")
        else:
            print(f"    ANOVA failed: {anova_result['error']}")
        
        # Perform Levene test
        levene_result = perform_levenes_test(run_eval_metrics_clean, metric)
        if 'error' not in levene_result:
            levene_result['metric'] = metric
            levene_results.append(levene_result)
            print(f"    Levene test completed - p_value: {levene_result['levene_p_value']:.4f}")
        else:
            print(f"    Levene test failed: {levene_result['error']}")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display the table
    print("\nBefore/After Comparison Table with Statistical Tests:")
    print("=" * 200)
    print("Columns show: Step 0, Step 550, 0→550 improvement (all models)")
    print("Step 50, 50→550 improvement (only for 0.5B models - larger models get formatting right from start)")
    print(comparison_df.to_string(index=False))
    
    # Display statistical summary
    if statistical_results:
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE SUMMARY")
        print("="*80)
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("0→550: Initial vs Final performance")
        print("50→550: Early vs Final performance (distinguishes formatting vs learning)")
        print()
        
        stats_df = pd.DataFrame(statistical_results)
        print(stats_df.to_string(index=False))
    
    # Display ANOVA results
    if anova_results:
        print("\n" + "="*80)
        print("ANOVA RESULTS - Testing Factors Influencing Improvement (Delta)")
        print("="*80)
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("Delta = Score_Step550 - Score_Step0")
        print()
        
        for result in anova_results:
            metric_name = result['metric'].replace('eval_', '').replace('/score_mean', '').replace('/rewards_mean', '')
            display_name = eval_env_name_mapping.get(metric_name, metric_name)
            print(f"Environment: {display_name}")
            print(f"  Algorithm effect: p = {result['Algorithm_p']:.4f}")
            print(f"  Model Size effect: p = {result['Model_Size_p']:.4f}")
            print(f"  Task effect: p = {result['Task_p']:.4f}")
            print(f"  Algorithm × Model Size: p = {result['Algorithm_Model_Size_p']:.4f}")
            print(f"  Algorithm × Task: p = {result['Algorithm_Task_p']:.4f}")
            print(f"  Model Size × Task: p = {result['Model_Size_Task_p']:.4f}")
            print(f"  Algorithm × Model Size × Task: p = {result['Algorithm_Model_Size_Task_p']:.4f}")
            print(f"  N observations: {result['n_observations']}")
            print()
    
    # Display Levene test results
    if levene_results:
        print("\n" + "="*80)
        print("LEVENE'S TEST RESULTS - Testing Variance Stability")
        print("="*80)
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print("Tests if variance at Step 550 is significantly different from Step 0")
        print()
        
        levene_summary = []
        for result in levene_results:
            metric_name = result['metric'].replace('eval_', '').replace('/score_mean', '').replace('/rewards_mean', '')
            display_name = eval_env_name_mapping.get(metric_name, metric_name)
            
            levene_summary.append({
                'Environment': display_name,
                'Levene_p': f"{result['levene_p_value']:.4f}",
                'Significance': result['levene_significance'],
                'Var_Step0': f"{result['step_0_variance']:.4f}",
                'Var_Step550': f"{result['step_550_variance']:.4f}",
                'N_Step0': result['n_step_0'],
                'N_Step550': result['n_step_550']
            })
        
        levene_df = pd.DataFrame(levene_summary)
        print(levene_df.to_string(index=False))
    
    # Save to CSV
    output_file = 'before_after_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\nTable saved to: {output_file}")
    
    # Save statistical results
    if statistical_results:
        stats_output_file = os.path.join(figures_dir, 'statistical_results.csv')
        stats_df.to_csv(stats_output_file, index=False)
        print(f"Statistical results saved to: {stats_output_file}")
    
    # Save ANOVA results
    if anova_results:
        anova_output_file = os.path.join(figures_dir, 'anova_results.csv')
        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(anova_output_file, index=False)
        print(f"ANOVA results saved to: {anova_output_file}")
    
    # Save Levene test results
    if levene_results:
        levene_output_file = os.path.join(figures_dir, 'levene_results.csv')
        levene_df = pd.DataFrame(levene_results)
        levene_df.to_csv(levene_output_file, index=False)
        print(f"Levene test results saved to: {levene_output_file}")
    
    return comparison_df, statistical_results, anova_results, levene_results

# Create the before/after comparison table


# Export to LaTeX
def export_to_latex(comparison_df, statistical_results, filename=os.path.join(figures_dir, 'before_after_comparison.tex'), use_score=False):
    """
    Export the comparison table to LaTeX format with proper formatting.
    Creates two separate tables: one for Snake tasks and one for FrozenLake tasks.
    Each table has separate columns for Algorithm (PPO/GRPO) and Model Size.
    Includes p-values and t-statistics from statistical_results.
    """
    print(f"\nExporting tables to LaTeX: {filename}")
    
    # Define metrics for each environment
    if use_score:
        snake_metrics = [
            'eval_FastSnake-Default/score_mean',
            'eval_Snake-20Step/score_mean'
        ]
        frozenlake_metrics = [
            'eval_FrozenLake-NoSlip/score_mean',
            'eval_FrozenLake-Slippery/score_mean'
        ]
    else:
        snake_metrics = [
            'eval_FastSnake-Default/rewards_mean',
            'eval_Snake-20Step/rewards_mean'
        ]
        frozenlake_metrics = [
            'eval_FrozenLake-NoSlip/rewards_mean',
            'eval_FrozenLake-Slippery/rewards_mean'
        ]
    
    # Get unique groups
    groups = run_eval_metrics_clean['group'].unique()
    target_groups = [g for g in groups if g not in ['FS_PPO_7B']]
    
    def get_statistical_data(group_name, metric_display_name):
        """Helper function to get statistical data for a specific group and metric"""
        # Find matching statistical result
        for result in statistical_results:
            if (result['Group'] == group_name and 
                result['Metric'] == metric_display_name and 
                result['Comparison'] == '0→550'):
                return result
        return None
    
    def create_table_data(metrics, env_name):
        """Helper function to create table data for a specific environment"""
        table_data = []
        
        this_target_groups = [g for g in target_groups if env_name.lower() in g.lower()]
        print(f"TARGET GROUPS: {this_target_groups}")

        # Group by algorithm first (PPO, then GRPO)
        ppo_groups = [g for g in this_target_groups if 'PPO' in g]
        grpo_groups = [g for g in this_target_groups if 'GRPO' in g]
        
        # Process PPO groups first
        for i, group in enumerate(ppo_groups):
            group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
            
            # Extract model size
            if '0pt5B' in group:
                model_size = '0.5B'
            elif '3B' in group:
                model_size = '3B'
            elif '7B' in group:
                model_size = '7B'
            else:
                model_size = 'Unknown'
            
            for j, metric in enumerate(metrics):
                # Format the metric name for display
                env_name_raw = metric.replace('eval_', '').replace('/score_mean', '').replace('/rewards_mean', '')
                display_name = eval_env_name_mapping.get(env_name_raw, env_name_raw)
                
                # Get data for step 0 and 550
                step_0_data = group_data[group_data['_step'] == 0][metric].dropna()
                step_550_data = group_data[group_data['_step'] == 550][metric].dropna()
                
                # Calculate means and stds
                step_0_mean = step_0_data.mean() if len(step_0_data) > 0 else np.nan
                step_0_std = step_0_data.std() if len(step_0_data) > 0 else np.nan
                step_550_mean = step_550_data.mean() if len(step_550_data) > 0 else np.nan
                step_550_std = step_550_data.std() if len(step_550_data) > 0 else np.nan
                
                # Calculate delta
                delta = step_550_mean - step_0_mean if not (np.isnan(step_550_mean) or np.isnan(step_0_mean)) else np.nan
                
                # Format the values for LaTeX
                if not np.isnan(step_0_mean):
                    step_0_formatted = f"${step_0_mean:.3f} \\pm {step_0_std:.3f}$"
                else:
                    step_0_formatted = "--"
                    
                if not np.isnan(step_550_mean):
                    step_550_formatted = f"${step_550_mean:.3f} \\pm {step_550_std:.3f}$"
                else:
                    step_550_formatted = "--"
                    
                if not np.isnan(delta):
                    delta_formatted = f"${delta:+.3f}$"
                else:
                    delta_formatted = "--"
                
                # Show algorithm name only for the first row of PPO
                algorithm_name = 'PPO' if (i == 0 and j == 0) else ''
                
                # Get statistical data
                mapped_group_name = run_name_mapping.get(group, group)
                stat_data = get_statistical_data(mapped_group_name, display_name)
                
                # Format statistical values
                if stat_data and not np.isnan(stat_data['p_value']):
                    p_value_formatted = f"${stat_data['p_value']:.4f}$"
                    t_stat_formatted = f"${stat_data['t_statistic']:.4f}$"
                    significance_formatted = f"$^{{{stat_data['Significant']}}}$"
                else:
                    p_value_formatted = "--"
                    t_stat_formatted = "--"
                    significance_formatted = "--"
                
                # Add row to table
                table_data.append({
                    'Algorithm': algorithm_name,
                    'Model Size': model_size,
                    'Task': display_name,
                    'Step 0': step_0_formatted,
                    'Step 550': step_550_formatted,
                    '$\\Delta$': delta_formatted,
                    'p-value': p_value_formatted,
                    't-stat': t_stat_formatted,
                    'Sig.': significance_formatted
                })
        
        # Process GRPO groups second
        for i, group in enumerate(grpo_groups):
            group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
            
            # Extract model size
            if '0pt5B' in group:
                model_size = '0.5B'
            elif '3B' in group:
                model_size = '3B'
            elif '7B' in group:
                model_size = '7B'
            else:
                model_size = 'Unknown'
            
            for j, metric in enumerate(metrics):
                # Format the metric name for display
                env_name_raw = metric.replace('eval_', '').replace('/score_mean', '').replace('/rewards_mean', '')
                display_name = eval_env_name_mapping.get(env_name_raw, env_name_raw)
                
                # Get data for step 0 and 550
                step_0_data = group_data[group_data['_step'] == 0][metric].dropna()
                step_550_data = group_data[group_data['_step'] == 550][metric].dropna()
                
                # Calculate means and stds
                step_0_mean = step_0_data.mean() if len(step_0_data) > 0 else np.nan
                step_0_std = step_0_data.std() if len(step_0_data) > 0 else np.nan
                step_550_mean = step_550_data.mean() if len(step_550_data) > 0 else np.nan
                step_550_std = step_550_data.std() if len(step_550_data) > 0 else np.nan
                
                # Calculate delta
                delta = step_550_mean - step_0_mean if not (np.isnan(step_550_mean) or np.isnan(step_0_mean)) else np.nan
                
                # Format the values for LaTeX
                if not np.isnan(step_0_mean):
                    step_0_formatted = f"${step_0_mean:.3f} \\pm {step_0_std:.3f}$"
                else:
                    step_0_formatted = "--"
                    
                if not np.isnan(step_550_mean):
                    step_550_formatted = f"${step_550_mean:.3f} \\pm {step_550_std:.3f}$"
                else:
                    step_550_formatted = "--"
                    
                if not np.isnan(delta):
                    delta_formatted = f"${delta:+.3f}$"
                else:
                    delta_formatted = "--"
                
                # Show algorithm name only for the first row of GRPO
                algorithm_name = 'GRPO' if (i == 0 and j == 0) else ''
                
                # Get statistical data
                mapped_group_name = run_name_mapping.get(group, group)
                stat_data = get_statistical_data(mapped_group_name, display_name)
                
                # Format statistical values
                if stat_data and not np.isnan(stat_data['p_value']):
                    p_value_formatted = f"${stat_data['p_value']:.4f}$"
                    t_stat_formatted = f"${stat_data['t_statistic']:.4f}$"
                    significance_formatted = f"$^{{{stat_data['Significant']}}}$"
                else:
                    p_value_formatted = "--"
                    t_stat_formatted = "--"
                    significance_formatted = "--"
                
                # Add row to table
                table_data.append({
                    'Algorithm': algorithm_name,
                    'Model Size': model_size,
                    'Task': display_name,
                    'Step 0': step_0_formatted,
                    'Step 550': step_550_formatted,
                    '$\\Delta$': delta_formatted,
                    'p-value': p_value_formatted,
                    't-stat': t_stat_formatted,
                    'Sig.': significance_formatted
                })
        
        return table_data
    
    # Create data for both tables
    snake_data = create_table_data(snake_metrics, 'FS')
    frozenlake_data = create_table_data(frozenlake_metrics, 'FL')
    
    # Create DataFrames
    snake_df = pd.DataFrame(snake_data)
    frozenlake_df = pd.DataFrame(frozenlake_data)
    
    # Generate LaTeX tables
    snake_latex = snake_df.to_latex(
        index=False,
        escape=False,
        caption="Snake Environment Performance: Initial vs Final (with Statistical Tests)",
        label="tab:snake_comparison",
        column_format='l' + 'l' + 'l' + 'c' * 6,  # l for text columns, c for numeric columns (3 original + 3 new)
        position='htbp'
    )
    
    frozenlake_latex = frozenlake_df.to_latex(
            index=False,
            escape=False,
        caption="FrozenLake Environment Performance: Initial vs Final (with Statistical Tests)",
        label="tab:frozenlake_comparison",
        column_format='l' + 'l' + 'l' + 'c' * 6,  # l for text columns, c for numeric columns (3 original + 3 new)
            position='htbp'
        )
    
    # Combine tables
    combined_latex = snake_latex + "\n\n" + frozenlake_latex
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(combined_latex)
    
    print(f"LaTeX tables saved to: {filename}")
    
    # Print previews
    print("\nSnake Table Preview:")
    print("=" * 80)
    print(snake_latex)
    
    print("\nFrozenLake Table Preview:")
    print("=" * 80)
    print(frozenlake_latex)
    
    return combined_latex

# Export the table to LaTeX
before_after_table, statistical_results, anova_results, levene_results = create_before_after_table(use_score=True)
latex_output = export_to_latex(before_after_table, statistical_results, os.path.join(figures_dir, 'before_after_comparison_SCORE.tex'), use_score=True)
before_after_table, statistical_results, anova_results, levene_results = create_before_after_table(use_score=False)
latex_output = export_to_latex(before_after_table, statistical_results, os.path.join(figures_dir, 'before_after_comparison_REWARDS.tex'), use_score=False)

# ------------------
# Create Bar Charts: Base LLM vs Finetuned Models
# ------------------

def create_bar_charts(use_score=True):
    """
    Create bar charts comparing base LLM models with post-trained models for each evaluation environment.
    Each chart shows one evaluation environment with separate bars for each model configuration.
    All data is loaded from full_llm_evals directory.
    """
    print("\n" + "="*60)
    print("CREATING BAR CHARTS: BASE LLM vs POST-TRAINED MODELS")
    print("="*60)
    
    # Load both base LLM and post-trained model evaluation data from full_llm_evals
    import json
    import os
    from pathlib import Path
    
    base_llm_data = []
    posttrained_llm_data = []
    bespoke_rl_data = []
    base_llm_dir = os.path.join(data_dir, 'full_llm_evals')
    # base_llm_dir = os.path.join(data_dir, 'full_llm_evals/poison_redos')
    
    # Load bespoke model data from specified directory
    bespoke_dir = os.path.join(base_llm_dir, 'snakebench_job_lr1em05_eps100000_g0p9_155795')
    if os.path.isdir(bespoke_dir):
        eval_results_file = os.path.join(bespoke_dir, 'eval_results.json')
        if os.path.exists(eval_results_file):
            try:
                with open(eval_results_file, 'r') as f:
                    data = json.load(f)
                
                # Process each environment in the results
                for env_name, env_data in data.items():
                    # Use pre-calculated values from JSON
                    score_mean_calc = env_data['mean_score']
                    score_std_calc = env_data['std_score']
                    score_95ci_calc = stats.t.interval(0.95, len(env_data['scores']) - 1, loc=score_mean_calc, scale=stats.sem(env_data['scores']))
                    reward_mean_calc = env_data['mean_reward']
                    reward_std_calc = env_data['std_reward']
                    reward_95ci_calc = stats.t.interval(0.95, len(env_data['rewards']) - 1, loc=reward_mean_calc, scale=stats.sem(env_data['rewards']))
                    
                    bespoke_rl_data.append({
                        'model_name': 'DQN',
                        'env': env_name,
                        'mean_score': score_mean_calc,
                        'std_score': score_std_calc,
                        '95ci_score': score_95ci_calc,
                        'mean_reward': reward_mean_calc,
                        'std_reward': reward_std_calc,
                        '95ci_reward': reward_95ci_calc,
                        'n_runs': 1
                    })
            except Exception as e:
                print(f"Error processing bespoke model {eval_results_file}: {e}")
    
    # Process all evaluation directories
    for subdir in os.listdir(base_llm_dir):
        subdir_path = os.path.join(base_llm_dir, subdir)
        if os.path.isdir(subdir_path):
            eval_results_file = os.path.join(subdir_path, 'eval_results.json')
            if os.path.exists(eval_results_file):
                try:
                    with open(eval_results_file, 'r') as f:
                        data = json.load(f)
                    
                    # Determine if this is a base model or post-trained model
                    # Post-trained: directories starting with FS_PPO or FL_PPO
                    # Base: directories starting with Qwen2.5
                    is_posttrained = subdir.startswith('FS_PPO') or subdir.startswith('FL_PPO')
                    is_base = subdir.startswith('Qwen2.5')
                    
                    # Skip if neither (e.g., old format subdirs)
                    if not is_posttrained and not is_base:
                        continue
                    
                    parts = subdir.split('_')
                    
                    if is_base:
                        # Base model format: Qwen2.5-{size}-Instruct_aamas_evals...
                        # Extract size from first part
                        model_size = parts[0].split('-')[1]  # Extract size (0.5B, 3B, 7B, etc.)
                        is_4096 = '4096' in subdir
                        is_cot = 'cot' in subdir
                        # Process each environment in the results
                        for env_name, env_data in data.items():

                            # Use pre-calculated values from JSON
                            score_mean_calc = env_data['mean_score']
                            score_std_calc = env_data['std_score']
                            score_95ci_calc = stats.t.interval(0.95, len(env_data['scores']) - 1, loc=score_mean_calc, scale=stats.sem(env_data['scores']))
                            reward_mean_calc = env_data['mean_reward']
                            reward_std_calc = env_data['std_reward']
                            reward_95ci_calc = stats.t.interval(0.95, len(env_data['rewards']) - 1, loc=reward_mean_calc, scale=stats.sem(env_data['rewards']))
                            
                            base_llm_data.append({
                                'model_size': model_size,
                                'is_4096': is_4096,
                                'is_cot': is_cot,
                                'env': env_name,
                                'mean_score': score_mean_calc,
                                'std_score': score_std_calc,
                                '95ci_score': score_95ci_calc,
                                'mean_reward': reward_mean_calc,
                                'std_reward': reward_std_calc,
                                '95ci_reward': reward_95ci_calc,
                                'n_runs': 1  # Each directory represents one run
                            })
                    
                    elif is_posttrained:
                        # Post-trained model format: FS_PPO_3B_run12_600_converted_aamas_evals...
                        # or FL_PPO_3B_run10_600_converted_aamas_evals...
                        # Extract model size and run info
                        model_size = parts[2]  # e.g., '3B', '7B', '14B'
                        
                        # Extract run number (e.g., 'run12' from parts[3])
                        run_part = parts[3] if len(parts) > 3 else 'run0'
                        
                        # Determine if this is FS (FastSnake) or FL (FrozenLake)
                        task_type = '' if subdir.startswith('FS_PPO') else 'FL'
                        
                        # Extract context and cot settings
                        is_4096 = '4096' in subdir
                        is_cot = 'cot' in subdir
                        
                        # Create a model name based on the configuration
                        context_str = '-4096' if is_4096 else '-256'
                        cot_str = '-CoT' if is_cot else ''
                        model_name = f"{task_type} PPO {model_size}{context_str}{cot_str}"

                        # Process each environment in the results
                        for env_name, env_data in data.items():
                            # Use pre-calculated values from JSON
                            score_mean_calc = env_data['mean_score']
                            score_std_calc = env_data['std_score']
                            score_95ci_calc = stats.t.interval(0.95, len(env_data['scores']) - 1, loc=score_mean_calc, scale=stats.sem(env_data['scores']))
                            reward_mean_calc = env_data['mean_reward']
                            reward_std_calc = env_data['std_reward']
                            reward_95ci_calc = stats.t.interval(0.95, len(env_data['rewards']) - 1, loc=reward_mean_calc, scale=stats.sem(env_data['rewards']))
                            
                            posttrained_llm_data.append({
                                'model_name': model_name,
                                'model_size': model_size,
                                'task_type': task_type,
                                'run_id': run_part,
                                'is_4096': is_4096,
                                'is_cot': is_cot,
                                'env': env_name,
                                'mean_score': score_mean_calc,
                                'std_score': score_std_calc,
                                '95ci_score': score_95ci_calc,
                                'mean_reward': reward_mean_calc,
                                'std_reward': reward_std_calc,
                                '95ci_reward': reward_95ci_calc,
                                'n_runs': 1  # Each directory represents one run
                            })
                except Exception as e:
                    print(f"Error processing {eval_results_file}: {e}")
    
    # Convert to DataFrames
    base_llm_df = pd.DataFrame(base_llm_data)
    posttrained_df = pd.DataFrame(posttrained_llm_data)
    bespoke_df = pd.DataFrame(bespoke_rl_data)
    
    if len(base_llm_df) == 0:
        print("No base LLM data found!")
        return
    
    print(f"Loaded {len(base_llm_df)} base LLM data points")
    print(f"Loaded {len(posttrained_df)} post-trained model data points")
    print(f"Loaded {len(bespoke_df)} bespoke model data points")
    
    # Aggregate base LLM data by model configuration and environment
    # base_llm_aggregated = []
    # for (model_size, is_4096, is_cot, env), group in base_llm_df.groupby(['model_size', 'is_4096', 'is_cot', 'env']):
    #     base_llm_aggregated.append({
    #         'model_size': model_size,
    #         'is_4096': is_4096,
    #         'is_cot': is_cot,
    #         'env': env,
    #         'mean_score': group['mean_score'].mean(),
    #         'std_score': group['mean_score'].std(),
    #         'mean_reward': group['mean_reward'].mean(),
    #         'std_reward': group['mean_reward'].std(),
    #         'n_runs': len(group)
    #     })
    
    # base_llm_df = pd.DataFrame(base_llm_aggregated)
    # print(f"Aggregated to {len(base_llm_df)} base LLM configurations")
    
    # Aggregate post-trained data by model configuration and environment
    # posttrained_aggregated = []
    # for (model_name, model_size, task_type, is_4096, is_cot, env), group in posttrained_df.groupby(['model_name', 'model_size', 'task_type', 'is_4096', 'is_cot', 'env']):
    #     posttrained_aggregated.append({
    #         'model_name': model_name,
    #         'model_size': model_size,
    #         'task_type': task_type,
    #         'is_4096': is_4096,
    #         'is_cot': is_cot,
    #         'env': env,
    #         'mean_score': group['mean_score'].mean(),
    #         'std_score': group['mean_score'].std(),
    #         'mean_reward': group['mean_reward'].mean(),
    #         'std_reward': group['mean_reward'].std(),
    #         'n_runs': len(group)
    #     })
    
    # posttrained_df = pd.DataFrame(posttrained_aggregated)
    # print(f"Aggregated to {len(posttrained_df)} post-trained configurations")
    
    # Define evaluation environments to include (excluding val and BigSlippery)
    target_envs = [
        'FastSnake-Default',
        'Snake-20Step', 
        'Snake-20Steps-PoisonAppleAndBanana',
        'FrozenLake-NoSlip',
        'FrozenLake-Slippery',
        'BabyAI-GoToRedBallNoDists',
        'BabyAI-GoToRedBall',
        'BabyAI-PickupDist'
    ]
    
    # finetuned_df already contains the aggregated post-trained data
    # No need to load from CSV anymore - all data comes from full_llm_evals!
    
    # Create bar chart for each environment
    for env in target_envs:
        print(f"\nCreating bar chart for {env}...")
        
        # Filter data for this environment
        env_base_data = base_llm_df[base_llm_df['env'] == env].copy()
        env_posttrained_data = posttrained_df[posttrained_df['env'] == env].copy()
        env_bespoke_data = bespoke_df[bespoke_df['env'] == env].copy() if len(bespoke_df) > 0 else None
        
        if len(env_base_data) == 0 and len(env_posttrained_data) == 0 and (env_bespoke_data is None or len(env_bespoke_data) == 0):
            print(f"No data found for {env}, skipping...")
            continue
        
        # Create model labels for base LLMs
        def create_base_llm_label(row):
            size = row['model_size']
            context = '4096' if row['is_4096'] else '256'
            cot = '-CoT' if row['is_cot'] else ''
            return f"{size}-{context}{cot}"
        
        env_base_data['label'] = env_base_data.apply(create_base_llm_label, axis=1)
        
        # Combine and sort data
        all_models = []
        
        # Add base LLM models - sort by model size first
        size_order = {'0.5B': 0, '3B': 1, '7B': 2, '14B': 3, '32B': 4, '72B': 5}
        for _, row in env_base_data.iterrows():
            all_models.append({
                'label': row['label'],
                'mean_score': row['mean_score'],
                'std_score': row['std_score'],
                '95ci_score': row['95ci_score'],
                'mean_reward': row['mean_reward'],
                'std_reward': row['std_reward'],
                '95ci_reward': row['95ci_reward'],
                'std_reward': row['std_reward'],
                'model_type': 'base_llm',
                'model_size': row['model_size'],
                'sort_key': (size_order.get(row['model_size'], 999), 0, row['is_4096'], row['is_cot'])
            })
        
        # Add post-trained models - model_size is directly available
        for _, row in env_posttrained_data.iterrows():
            # Model size is already extracted
            model_size = row['model_size']
            task_type = row['task_type']

            # Create sort key: Frozen Lake first, then 0.5B before 3B
            env_priority = 0 if task_type == 'FL' else 1  # Frozen Lake first

            if model_size == '0.5B':
                size_priority = 0
            elif model_size == '3B':
                size_priority = 1
            elif model_size == '7B':
                size_priority = 2
            elif model_size == '14B':
                size_priority = 3
            else:
                size_priority = 999

            all_models.append({
                'label': row['model_name'],
                'mean_score': row['mean_score'],
                'std_score': row['std_score'],
                '95ci_score': row['95ci_score'],
                'mean_reward': row['mean_reward'],
                'std_reward': row['std_reward'],
                '95ci_reward': row['95ci_reward'],
                'model_type': 'posttrained',
                'model_size': model_size,
                'sort_key': (size_priority, 1, env_priority, row['model_name'])
            })

        # Add bespoke model data (DQN) if available for this environment
        # if env_bespoke_data is not None:
        #     for _, row in env_bespoke_data.iterrows():
        #         all_models.append({
        #             'label': row['model_name'],
        #             'mean_score': row['mean_score'],
        #             'std_score': row['std_score'],
        #             '95ci_score': row['95ci_score'],
        #             'mean_reward': row['mean_reward'],
        #             'std_reward': row['std_reward'],
        #             '95ci_reward': row['95ci_reward'],
        #             'model_type': 'bespoke',
        #             'model_size': 'DQN',
        #             'sort_key': (999, 999, 999, 999, 'bespoke')  # Sort bespoke models last
        #         })
        
        # Add manual entry for FS GRPO 3B-256 for Snake-20Step environment

        grpo_mean_score = 1.28
        grpo_std_score = 1.05
        grpo_n = 100
        grpo_sem_score = grpo_std_score / (grpo_n ** 0.5)
        grpo_ci_score = stats.t.interval(0.95, df=grpo_n-1, loc=grpo_mean_score, scale=grpo_sem_score)
        grpo_mean_reward = 0.44
        grpo_std_reward = 0.99
        grpo_n = 100
        grpo_sem_reward = grpo_std_reward / (grpo_n ** 0.5)
        grpo_ci_reward = stats.t.interval(0.95, df=grpo_n-1, loc=grpo_mean_reward, scale=grpo_sem_reward)
        if env == 'Snake-20Step':
            all_models.append({
                'label': 'GRPO 3B-256',
                'mean_score': grpo_mean_score,
                'std_score': grpo_std_score,  # No std provided
                '95ci_score': grpo_ci_score,
                'mean_reward': grpo_mean_reward,
                '95ci_reward': grpo_ci_reward,  
                'std_reward': grpo_std_reward,
                'model_type': 'posttrained',
                'model_size': '3B', 
                'sort_key': (1, 1, 1, 'GRPO 3B-256')  # Sort with other 3B post-trained models
            })
        # grpo_mean_score = 1.28
        # grpo_std_score = 1.05
        # grpo_n = 100
        # grpo_sem_score = grpo_std_score / (grpo_n ** 0.5)
        # grpo_ci_score = stats.t.interval(0.95, df=grpo_n-1, loc=grpo_mean_score, scale=grpo_sem_score)
        # grpo_mean_reward = 0.44
        # grpo_std_reward = 0.99
        # grpo_n = 100
        # grpo_sem_reward = grpo_std_reward / (grpo_n ** 0.5)
        # if env == 'Snake-20Steps-PoisonAppleAndBanana':
        #     all_models.append({
        #         'label': 'GRPO 3B-256',
        #         'mean_score': 0.42,
        #         'std_score': 1.05,  # No std provided
        #         '95ci_score': (0.42,0.42),
        #         'mean_reward': -1.34,
        #         'std_reward': 0.99,
        #         '95ci_reward': (-1.34, -1.34),
        #         'model_type': 'posttrained',
        #         'model_size': '3B',
        #         'sort_key': (1, 1, 1, 'GRPO 3B-256')  # Sort with other 3B post-trained models
        #     })
        
        # Sort models: base LLMs first (by size, then config), then finetuned models
        all_models.sort(key=lambda x: x['sort_key'])
        
        if len(all_models) == 0:
            print(f"No data to plot for {env}")
            continue
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(5, 4))  # Wide figure for many bars
        
        labels = [model['label'] for model in all_models]
        scores = [model['mean_score'] for model in all_models]
        rewards = [model['mean_reward'] for model in all_models]
        errors_score = [model['std_score'] for model in all_models]
        errors_reward = [model['std_reward'] for model in all_models]
        ci95_score = [model['95ci_score'] for model in all_models]
        ci95_reward = [model['95ci_reward'] for model in all_models]
        
        # Color by model size for base models, pink for all finetuned models
        # Matching scatter chart colors from base_llm_comparisons.ipynb
        size_colors = {
            '0.5B': '#949494',  # Grey
            '3B': '#D55E00',    # Orange  
            '7B': '#029E73',   # Green
            '14B': '#DE8F05', # Yellow/Orange
            '32B': '#FBAFE4',    # Pink
            '72B': '#0173B2',   # Blue
            'Unknown': '#000000' # Black fallback
        }
        
        # Color models by size, with base models having distinctive hatch patterns
        colors = []
        hatches = []
        bespoke_color = '#000000'  # Black for bespoke models (DQN)
        for model in all_models:
            if model['model_type'] == 'base_llm':
                colors.append(size_colors.get(model['model_size'], '#949494'))
                hatches.append('')  # Dense cross hatch for base models
            elif model['model_type'] == 'bespoke':
                colors.append(bespoke_color)  # Black for bespoke models
                hatches.append('')  # No hatch for bespoke models
            else:  # posttrained models - use same color as their model size
                colors.append(size_colors.get(model['model_size'], '#949494'))
                hatches.append('xxx')  # No hatch for post-trained models
        
        # Create bars - separate base models (no error bars) and post-trained models (with error bars)
        x_pos = range(len(labels))
        
        # Split into base, posttrained, and bespoke models
        base_indices = [i for i, model in enumerate(all_models) if model['model_type'] == 'base_llm']
        posttrained_indices = [i for i, model in enumerate(all_models) if model['model_type'] == 'posttrained']
        bespoke_indices = [i for i, model in enumerate(all_models) if model['model_type'] == 'bespoke']
        
        # Plot base models without error bars (with hatch patterns)
        if base_indices:
            base_x = [x_pos[i] for i in base_indices]
            base_scores = [scores[i] for i in base_indices]
            base_rewards = [rewards[i] for i in base_indices]
            base_ci95_score = [(scores[i] - ci95_score[i][0], ci95_score[i][1] - scores[i]) for i in base_indices]
            base_ci95_reward = [(rewards[i] - ci95_reward[i][0],  ci95_reward[i][1] - rewards[i]) for i in base_indices]
            base_colors = [colors[i] for i in base_indices]
            base_hatches = [hatches[i] for i in base_indices]
            ax.bar(base_x, base_scores if use_score else base_rewards, color=base_colors, hatch=base_hatches, alpha=0.7, width=0.45, yerr=np.array(base_ci95_score).transpose() if use_score else np.array(base_ci95_reward).transpose())
        
        # Plot post-trained models with error bars (no hatch)
        if posttrained_indices:
            posttrained_x = [x_pos[i] for i in posttrained_indices]
            posttrained_scores = [scores[i] for i in posttrained_indices]
            posttrained_rewards = [rewards[i] for i in posttrained_indices]
            posttrained_errors_score = [errors_score[i] for i in posttrained_indices]
            posttrained_errors_reward = [errors_reward[i] for i in posttrained_indices]
            posttrained_colors = [colors[i] for i in posttrained_indices]
            posttrained_hatches = [hatches[i] for i in posttrained_indices]
            posttrained_ci95_score = [(scores[i] - ci95_score[i][0], ci95_score[i][1] - scores[i]) for i in posttrained_indices]
            posttrained_ci95_reward = [(rewards[i] - ci95_reward[i][0], ci95_reward[i][1] - rewards[i]) for i in posttrained_indices]
            print(np.array(posttrained_ci95_score).transpose())
            ax.bar(posttrained_x, posttrained_scores if use_score else posttrained_rewards, yerr=np.array(posttrained_ci95_score).transpose() if use_score else np.array(posttrained_ci95_reward).transpose(), 
            capsize=3,
                   color=posttrained_colors, hatch=posttrained_hatches, alpha=0.7, width=0.45)

        # Plot bespoke models with error bars (no hatch)
        if bespoke_indices:
            bespoke_x = [x_pos[i] for i in bespoke_indices]
            bespoke_scores = [scores[i] for i in bespoke_indices]
            bespoke_rewards = [rewards[i] for i in bespoke_indices]
            bespoke_ci95_score = [(scores[i] - ci95_score[i][0], ci95_score[i][1] - scores[i]) for i in bespoke_indices]
            bespoke_ci95_reward = [(rewards[i] - ci95_reward[i][0], ci95_reward[i][1] - rewards[i]) for i in bespoke_indices]
            bespoke_colors = [colors[i] for i in bespoke_indices]
            bespoke_hatches = [hatches[i] for i in bespoke_indices]
            ax.bar(bespoke_x, bespoke_scores if use_score else bespoke_rewards, 
                   yerr=np.array(bespoke_ci95_score).transpose() if use_score else np.array(bespoke_ci95_reward).transpose(),
                   capsize=3,
                   color=bespoke_colors, hatch=bespoke_hatches, alpha=0.7, width=0.45)
        
        # Customize the plot
        # ax.set_xlabel('Model Configuration', fontsize=12)
        ax.set_ylabel('Mean Score' if use_score else 'Mean Reward', fontsize=12)
        # ax.set_ylim(top=5.8)
        # ax.set_title(f'Model Performance Comparison: {eval_env_name_mapping.get(env, env)}', fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create 2-column legend showing model sizes and training types
        from matplotlib.patches import Patch
        legend_elements = []

        # Column 1: Model sizes
        legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Model Sizes:', linewidth=0))
        for size, color in size_colors.items():
            if size in [model['model_size'] for model in all_models]:
                legend_elements.append(Patch(facecolor=color, alpha=0.7, label=f'  {size}'))

        # Column 2: Training types and bespoke models
        legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Training:', linewidth=0))
        legend_elements.append(Patch(facecolor='lightgray', alpha=0.7, label='  Base'))
        legend_elements.append(Patch(facecolor='lightgray', hatch='xxx', alpha=0.7, label='  Post-trained'))

        if any(model['model_type'] == 'bespoke' for model in all_models):
            legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Other:', linewidth=0))
            legend_elements.append(Patch(facecolor=bespoke_color, alpha=0.7, label='  DQN'))
        
        # ax.legend(handles=legend_elements, loc='best', ncol=2, fontsize=9.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        save_name = f'bar_chart_{env.replace("-", "_").replace(" ", "_").lower()}_{"score" if use_score else "reward"}'
        save_path = os.path.join(figures_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart as {save_path}")
        
        plt.close()  # Close the figure to free memory
        print(all_models)

# ------------------
# Create Numerical Comparison Table: Base LLM vs Finetuned Models
# ------------------

def create_numerical_comparison_table():
    """
    Create a comprehensive table comparing numerical performance values of base LLM models
    with finetuned models for each evaluation environment. This table will include all
    the numerical data used in the bar charts.
    """
    print("\n" + "="*60)
    print("CREATING NUMERICAL COMPARISON TABLE: BASE LLM vs FINETUNED MODELS")
    print("="*60)

    # Load base LLM evaluation data (reuse logic from create_bar_charts)
    import json
    import os
    from pathlib import Path

    base_llm_data = []
    base_llm_dir = os.path.join(data_dir, 'full_llm_evals')

    # Process all base LLM evaluation directories
    for subdir in os.listdir(base_llm_dir):
        subdir_path = os.path.join(base_llm_dir, subdir)
        if os.path.isdir(subdir_path):
            eval_results_file = os.path.join(subdir_path, 'eval_results.json')
            if os.path.exists(eval_results_file):
                try:
                    with open(eval_results_file, 'r') as f:
                        data = json.load(f)

                    # Extract model configuration from directory name
                    # Format 1: Qwen2.5-{size}-Instruct_aamas_evals_{context}_{cot}_{timestamp}
                    # Format 2: FS_PPO_7B_run41_600_converted_aamas_evals_{context}_{cot}_{timestamp}
                    parts = subdir.split('_')

                    # Skip finetuned base models
                    if 'FLPPO' in subdir or 'FSPPO' in subdir:
                        continue
                    
                    # Detect format and extract model size
                    if 'Qwen' in subdir or 'Instruct' in subdir:
                        # Old format: Qwen2.5-{size}-Instruct
                        model_size = parts[0].split('-')[1]  # Extract size (0.5B, 3B, etc.)
                    elif 'FS_PPO' in subdir or 'FL_PPO' in subdir or 'FS_GRPO' in subdir or 'FL_GRPO' in subdir:
                        # New format: FS_PPO_7B or FL_PPO_14B, etc.
                        # Extract size from second part (e.g., '7B' from 'FS_PPO_7B')
                        model_size = parts[2] if len(parts) > 2 else 'Unknown'
                    else:
                        # Fallback: try to extract from first part
                        model_size = parts[0].split('-')[1] if '-' in parts[0] else 'Unknown'
                    
                    is_4096 = '4096' in subdir
                    is_cot = 'cot' in subdir

                    # Skip if we couldn't determine model size
                    if model_size == 'Unknown':
                        print(f"Warning: Could not determine model size for {subdir}, skipping")
                        continue

                    # Process each environment in the results
                    for env_name, env_data in data.items():
                        base_llm_data.append({
                            'model_size': model_size,
                            'is_4096': is_4096,
                            'is_cot': is_cot,
                            'env': env_name,
                            'mean_score': env_data.get('mean_score', 0.0),
                            'std_score': env_data.get('std_score', 0.0),
                            'n_runs': 1  # Each directory represents one run
                        })
                except Exception as e:
                    print(f"Error processing {eval_results_file}: {e}")

    # Convert to DataFrame
    base_llm_df = pd.DataFrame(base_llm_data)

    if len(base_llm_df) == 0:
        print("No base LLM data found!")
        return None, None

    print(f"Loaded {len(base_llm_df)} base LLM data points")

    # Aggregate base LLM data by model configuration and environment
    base_llm_aggregated = []
    for (model_size, is_4096, is_cot, env), group in base_llm_df.groupby(['model_size', 'is_4096', 'is_cot', 'env']):
        base_llm_aggregated.append({
            'model_size': model_size,
            'is_4096': is_4096,
            'is_cot': is_cot,
            'env': env,
            'mean_score': group['mean_score'].mean(),
            'std_score': group['mean_score'].std(),
            'n_runs': len(group)
        })

    base_llm_df = pd.DataFrame(base_llm_aggregated)
    print(f"Aggregated to {len(base_llm_df)} base LLM configurations")

    # Define evaluation environments to include
    target_envs = [
        'FastSnake-Default',
        'Snake-20Step',
        'Snake-20Steps-PoisonAppleAndBanana',
        'FrozenLake-NoSlip',
        'FrozenLake-Slippery',
        'BabyAI-GoToRedBallNoDists',
        'BabyAI-GoToRedBall',
        'BabyAI-PickupDist'
    ]

    # Get finetuned model data (final performance at step 550)
    finetuned_data = []
    for group in run_eval_metrics_clean['group'].unique():
        if group == 'FS_PPO_7B':  # Exclude 7B as requested
            continue

        group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
        final_data = group_data[group_data['_step'] == 550]

        if len(final_data) == 0:
            continue

        # Get mapped group name
        mapped_name = run_name_mapping.get(group, group)

        for env in target_envs:
            metric_col = f'eval_{env}/score_mean'
            if metric_col in final_data.columns:
                scores = final_data[metric_col].dropna()
                if len(scores) > 0:
                    finetuned_data.append({
                        'model_name': mapped_name,
                        'group': group,
                        'env': env,
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'n_runs': len(scores)
                    })

    # Add best individual runs for FL and FS PPO 3B
    best_runs_data = []

    # Best FS PPO 3B run (run 12)
    fs_ppo_3b_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == 'FS_PPO_3B']
    fs_ppo_3b_run_12 = fs_ppo_3b_data[
        fs_ppo_3b_data['experiment_name'].str.split('_').str[-2].str.endswith('run12', na=False)
    ]
    if len(fs_ppo_3b_run_12) > 0:
        final_data = fs_ppo_3b_run_12[fs_ppo_3b_run_12['_step'] == 550]
        if len(final_data) > 0:
            for env in target_envs:
                metric_col = f'eval_{env}/score_mean'
                if metric_col in final_data.columns:
                    scores = final_data[metric_col].dropna()
                    if len(scores) > 0:
                        best_runs_data.append({
                            'model_name': 'Snake PPO 3B (Best)',
                            'group': 'FS_PPO_3B_best',
                            'env': env,
                            'mean_score': scores.mean(),
                            'std_score': 0.0,  # Single run, no std
                            'n_runs': 1
                        })

    # Best FL PPO 3B run (run 10)
    fl_ppo_3b_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == 'FL_PPO_3B']
    fl_ppo_3b_run_10 = fl_ppo_3b_data[
        fl_ppo_3b_data['experiment_name'].str.split('_').str[-2].str.endswith('run11', na=False)
    ]
    if len(fl_ppo_3b_run_10) > 0:
        final_data = fl_ppo_3b_run_10[fl_ppo_3b_run_10['_step'] == 550]
        if len(final_data) > 0:
            for env in target_envs:
                metric_col = f'eval_{env}/score_mean'
                if metric_col in final_data.columns:
                    scores = final_data[metric_col].dropna()
                    if len(scores) > 0:
                        best_runs_data.append({
                            'model_name': 'Frozen Lake PPO 3B (Best)',
                            'group': 'FL_PPO_3B_best',
                            'env': env,
                            'mean_score': scores.mean(),
                            'std_score': 0.0,  # Single run, no std
                            'n_runs': 1
                        })

    # Combine all finetuned data
    finetuned_data.extend(best_runs_data)
    finetuned_df = pd.DataFrame(finetuned_data)
    print(f"Loaded {len(finetuned_df)} finetuned model data points")

    # Create comprehensive table data
    table_data = []

    for env in target_envs:
        # Filter data for this environment
        env_base_data = base_llm_df[base_llm_df['env'] == env].copy()
        env_finetuned_data = finetuned_df[finetuned_df['env'] == env].copy()

        if len(env_base_data) == 0 and len(env_finetuned_data) == 0:
            continue

        # Create model labels for base LLMs
        def create_base_llm_label(row):
            size = row['model_size']
            context = '4096' if row['is_4096'] else '256'
            cot = '-CoT' if row['is_cot'] else ''
            return f"{size}-{context}{cot}"

        env_base_data['label'] = env_base_data.apply(create_base_llm_label, axis=1)

        # Sort base models by size and configuration
        size_order = {'0.5B': 0, '3B': 1, '32B': 2, '72B': 3}
        env_base_data['sort_key'] = env_base_data['model_size'].map(size_order)

        # Add base LLM data to table
        for _, row in env_base_data.sort_values('sort_key').iterrows():
            table_data.append({
                'Environment': eval_env_name_mapping.get(env, env),
                'Model_Type': 'Base LLM',
                'Model_Name': row['label'],
                'Model_Size': row['model_size'],
                'Mean_Score': f"{row['mean_score']:.3f}",
                'Std_Score': f"{row['std_score']:.3f}"
            })

        # Add finetuned model data to table
        for _, row in env_finetuned_data.iterrows():
            # Extract model size from group name for sorting
            group = row['group']
            if '0pt5B' in group:
                model_size = '0.5B'
            elif '3B' in group:
                model_size = '3B'
            elif '7B' in group:
                model_size = '7B'
            else:
                model_size = 'Unknown'

            table_data.append({
                'Environment': eval_env_name_mapping.get(env, env),
                'Model_Type': 'Finetuned',
                'Model_Name': row['model_name'],
                'Model_Size': model_size,
                'Mean_Score': f"{row['mean_score']:.3f}",
                'Std_Score': f"{row['std_score']:.3f}"
            })

    # Create DataFrame
    comparison_df = pd.DataFrame(table_data)

    if len(comparison_df) == 0:
        print("No data to create table!")
        return None, None

    # Display the table
    print("\nNumerical Comparison Table: Base LLM vs Finetuned Models")
    print("=" * 120)
    print(comparison_df.to_string(index=False))

    # Save to CSV
    output_file = os.path.join(figures_dir, 'numerical_comparison_base_vs_finetuned.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"\nNumerical comparison table saved to: {output_file}")

    return comparison_df, finetuned_df

# ------------------
# Export Numerical Comparison Table to LaTeX
# ------------------

def export_numerical_comparison_to_latex(comparison_df, filename=os.path.join(figures_dir, 'numerical_comparison_base_vs_finetuned.tex')):
    """
    Export the numerical comparison table to LaTeX format with proper formatting.
    Groups models by environment and separates base LLMs from finetuned models.
    """
    print(f"\nExporting numerical comparison table to LaTeX: {filename}")

    if comparison_df is None or len(comparison_df) == 0:
        print("No data to export!")
        return None

    # Group by environment
    environments = comparison_df['Environment'].unique()

    latex_sections = []

    for env in environments:
        env_data = comparison_df[comparison_df['Environment'] == env].copy()

        # Separate base LLMs and finetuned models
        base_models = env_data[env_data['Model_Type'] == 'Base LLM']
        finetuned_models = env_data[env_data['Model_Type'] == 'Finetuned']

        # Sort base models by size
        size_order = {'0.5B': 0, '3B': 1, '32B': 2, '72B': 3}
        base_models['sort_key'] = base_models['Model_Size'].map(size_order)
        base_models = base_models.sort_values('sort_key')

        # Sort finetuned models by model size and name
        finetuned_models = finetuned_models.sort_values(['Model_Size', 'Model_Name'])

        # Combine: base models first, then finetuned models
        combined_data = pd.concat([base_models, finetuned_models], ignore_index=True)

        # Create LaTeX table for this environment
        env_clean = env.replace(" ", "").replace("-", "")
        table_label = f"tab:{env_clean.lower()}"

        # Create table content
        latex_content = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Comparison: {env}}}
\\label{{{table_label}}}
\\begin{{tabular}}{{@{{}}l@{{}}l@{{}}c@{{}}c@{{}}}}
\\toprule
\\textbf{{Model Type}} & \\textbf{{Model Name}} & \\textbf{{Mean}} & \\textbf{{Std}} \\\\
\\midrule
"""

        for _, row in combined_data.iterrows():
            latex_content += f"{row['Model_Type']} & {row['Model_Name']} & ${row['Mean_Score']}$ & ${row['Std_Score']}$ \\\\\n"

        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""

        latex_sections.append(latex_content)

    # Combine all sections
    combined_latex = "\n\n".join(latex_sections)

    # Save to file
    with open(filename, 'w') as f:
        f.write(combined_latex)

    print(f"LaTeX table saved to: {filename}")

    # Print preview
    print("\nLaTeX Table Preview:")
    print("=" * 80)
    print(combined_latex[:1000] + "..." if len(combined_latex) > 1000 else combined_latex)

    return combined_latex

# Create the numerical comparison table and export to LaTeX
numerical_table, finetuned_df = create_numerical_comparison_table()
if numerical_table is not None:
    latex_output = export_numerical_comparison_to_latex(numerical_table)

# Create the bar charts
create_bar_charts(use_score=True)
create_bar_charts(use_score=False)
