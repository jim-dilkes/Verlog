import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats

from analysis_configs import legend_kwargs, run_name_mapping, run_env_mapping, run_alg_mapping, run_name_colors, eval_env_name_mapping

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


# ------------------
# Load the data
# ------------------


# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(current_dir, 'analysis/data')
figures_dir = os.path.join(current_dir, 'analysis/charts')
os.makedirs(figures_dir, exist_ok=True) 
# Load the data
run_eval_metrics = pd.read_csv(os.path.join(data_dir, "eval_metrics.csv"))
run_configs = pd.read_csv(os.path.join(data_dir, "run_configs.csv"))

print(f"Loading data from {data_dir}")
print(f"Evaluation metrics file: {os.path.join(data_dir, 'eval_metrics.csv')}")
print(f"Run configs file: {os.path.join(data_dir, 'run_configs.csv')}")

# Filter to only keep gpu from run configs
# gpu_choice = "NVIDIA H100 80GB HBM3"

# Merge configurations with metrics
run_eval_metrics = run_eval_metrics.merge(run_configs, on="run_id", how="inner")

# run_eval_metrics = run_eval_metrics[run_eval_metrics['run_id'].isin(group)]

# Display available columns
# print("Available metrics columns:")
# print([col for col in run_eval_metrics.columns if col.startswith('eval_')])
# print("\nAvailable config columns:")
# print([col for col in run_eval_metrics.columns if not col.startswith('eval_')])


# ------------------
# Clean the data
# ------------------


#  run_eval_metrics.columns:
# Keep columns starting with eval_, val/,
# also keep  _step, run_id, group, trainer_value_experiment_name

cols_to_keep = [col for col in run_eval_metrics.columns if col.startswith('eval_') or col.startswith('val/')]
cols_to_keep.extend(['_step', 'run_id', 'group', 'trainer_value_experiment_name'])
cols_to_keep = list(set(cols_to_keep))
cols_to_keep.sort()
run_eval_metrics_clean = run_eval_metrics[cols_to_keep]

# Keep only steps that are multiples of 50
run_eval_metrics_clean = run_eval_metrics_clean[run_eval_metrics_clean['_step'] % 50 == 0]

# Create global_experiment_name column - it should remove the final part of the name after the last _,but only if that last part starts with "slurm"
run_eval_metrics_clean['experiment_name'] = run_eval_metrics_clean['trainer_value_experiment_name'].str.replace(r'_slurm\d+$', '', regex=True)
run_eval_metrics_clean['slurm_job_id'] = (
    run_eval_metrics_clean['trainer_value_experiment_name']
    .str.extract(r'_slurm_?(\d+)', expand=False)
)
# 2. Fill NaN values (where the pattern wasn't found) with '0' (as a string initially)
run_eval_metrics_clean['slurm_job_id'] = (
    run_eval_metrics_clean['slurm_job_id'].fillna('0')
)
# 3. Convert the column to an integer type
run_eval_metrics_clean['slurm_job_id'] = (
    run_eval_metrics_clean['slurm_job_id'].astype(np.int64)
)
run_eval_metrics_clean.loc[
    run_eval_metrics_clean['run_id'] == '32xgf4aq', 
    'slurm_job_id'
] = 1


# Subtract 100 from _step for run_id==hzkacmiv
run_eval_metrics_clean.loc[
    run_eval_metrics_clean['run_id'] == 'hzkacmiv', 
    '_step'
] = run_eval_metrics_clean.loc[
    run_eval_metrics_clean['run_id'] == 'hzkacmiv', 
    '_step'
] - 100

print("Pre de-duplication number of rows: ", len(run_eval_metrics_clean))
# 1. Sort the data: This is the crucial step. We sort by the criteria that define a duplicate 
#    group, and then by the column we want to keep (slurm_job_id) in descending order.
run_eval_metrics_clean = run_eval_metrics_clean.sort_values(
    by=['experiment_name', '_step', 'slurm_job_id'],
    ascending=[True, True, True]  # Keep the largest slurm_job_id at the bottom
)

print(run_eval_metrics_clean.head())

# 2. Drop duplicates: Use 'keep='last' to retain the row that appeared last in the sort,
#    which is the one with the largest slurm_job_id.
# BUT: Do NOT remove duplicates for experiments with "0pt5B" in the name
# First, identify which experiments have "0pt5B" in the name
zero_pt5b_experiments = run_eval_metrics_clean[
    run_eval_metrics_clean['experiment_name'].str.contains('0pt5B', na=False)
]['experiment_name'].unique()

print(f"Found {len(zero_pt5b_experiments)} experiments with '0pt5B' in name - these will NOT be deduplicated:")
for exp in zero_pt5b_experiments:
    print(f"  - {exp}")

# Split data into two parts: 0pt5B experiments (no deduplication) and others (deduplication)
zero_pt5b_data = run_eval_metrics_clean[
    run_eval_metrics_clean['experiment_name'].str.contains('0pt5B', na=False)
]
other_data = run_eval_metrics_clean[
    ~run_eval_metrics_clean['experiment_name'].str.contains('0pt5B', na=False)
]

print(f"0pt5B experiments: {len(zero_pt5b_data)} rows (no deduplication)")
print(f"Other experiments: {len(other_data)} rows (will be deduplicated)")

# Deduplicate only the non-0pt5B experiments
if len(other_data) > 0:
    other_data_deduped = other_data.drop_duplicates(
        subset=['experiment_name', '_step'], 
        keep='last'
    ).reset_index(drop=True)
    print(f"After deduplication of other experiments: {len(other_data_deduped)} rows")
else:
    other_data_deduped = other_data

# Combine the two datasets back together
run_eval_metrics_clean = pd.concat([zero_pt5b_data, other_data_deduped], ignore_index=True)

# Add run numbers to experiment names that don't have them
print(f"\nAdding run numbers to experiment names...")

# Find experiment names that don't end with _x pattern (where x is a number)
# Pattern: experiment name should end with _ followed by digits
experiments_needing_numbers = run_eval_metrics_clean[
    ~run_eval_metrics_clean['experiment_name'].str.contains(r'_\d+$', regex=True, na=False)
]['experiment_name'].unique()

print(f"Found {len(experiments_needing_numbers)} experiment names that need run numbers:")
for exp in experiments_needing_numbers:
    print(f"  - {exp}")

# Create a mapping of run_id to run number (starting from 10000)
run_id_to_number = {}
current_number = 10000

# Get unique run_ids that have experiments needing numbers
run_ids_needing_numbers = run_eval_metrics_clean[
    run_eval_metrics_clean['experiment_name'].isin(experiments_needing_numbers)
]['run_id'].unique()

# Assign numbers to each run_id
for run_id in sorted(run_ids_needing_numbers):
    run_id_to_number[run_id] = current_number
    current_number += 1

print(f"\nAssigned run numbers to run_ids:")
for run_id, number in run_id_to_number.items():
    print(f"  - {run_id}: {number}")

# Apply the run numbers to experiment names
for run_id, number in run_id_to_number.items():
    # Find rows with this run_id and experiment names that need numbers
    mask = (run_eval_metrics_clean['run_id'] == run_id) & \
           (run_eval_metrics_clean['experiment_name'].isin(experiments_needing_numbers))
    
    # Update the experiment_name by appending the run number
    run_eval_metrics_clean.loc[mask, 'experiment_name'] = \
        run_eval_metrics_clean.loc[mask, 'experiment_name'] + f'_{number}'

print(f"\nRun number assignment complete.")

# --- Optional: Verification (You can remove the duplicate_entries section now) ---

print(f"\nDe-duplication complete.")
print(f"Final number of unique rows: {len(run_eval_metrics_clean)}")


# # Identify Look for duplicate entries - i.e. same experiment_name, same _step
duplicate_entries = run_eval_metrics_clean[run_eval_metrics_clean.duplicated(subset=['experiment_name', '_step'],keep=False)]
duplicate_entries[['experiment_name', '_step', 'trainer_value_experiment_name', 'group', 'run_id']].sort_values(by=['experiment_name', '_step'])


check_exp_per_group = run_eval_metrics_clean.groupby('group')['experiment_name'].nunique()
print(check_exp_per_group)



# ------------------
# Create training curves
# ------------------

# Create training curves for each rewards_mean metric
import matplotlib.pyplot as plt
import numpy as np

# Get all rewards_mean columns (exclude val/, BigSlippery, and BabyAI)
rewards_mean_cols = [col for col in run_eval_metrics_clean.columns 
                    if col.endswith('/rewards_mean') 
                    and not col.startswith('val/')
                    and 'BigSlippery' not in col
                    and 'BabyAI' not in col]
print(f"Found {len(rewards_mean_cols)} rewards_mean metrics:")
for col in rewards_mean_cols:
    print(f"  - {col}")

# Get unique groups and apply name mapping
groups = run_eval_metrics_clean['group'].unique()
print(f"\nFound {len(groups)} groups: {groups}")

# Function to create training curves for a specific environment
def create_training_curves(env_filter, title_suffix, save_name=None):
    # Filter rewards_mean columns for this environment
    if env_filter == 'FrozenLake':
        # For FL trained runs, show ALL evaluation environments
        env_cols = rewards_mean_cols
    elif env_filter == 'FastSnake':
        # For FS trained runs, show ALL evaluation environments  
        env_cols = rewards_mean_cols
    else:
        env_cols = rewards_mean_cols
    
    if len(env_cols) == 0:
        print(f"No columns found for {env_filter}")
        return
    
    # Filter groups for this environment
    if env_filter == 'FrozenLake':
        env_groups = [g for g in groups if g.startswith('FL_')]
    elif env_filter == 'FastSnake':
        env_groups = [g for g in groups if g.startswith('FS_')]
    else:
        env_groups = groups
    
    print(f"\n{env_filter} environment - Found {len(env_cols)} metrics and {len(env_groups)} groups")
    
    # Define the specific order of environments we want
    desired_order = [
        'eval_FastSnake-Default/rewards_mean',           # Snake -8
        'eval_Snake-20Step/rewards_mean',                # Snake -20
        'eval_Snake-20Steps-PoisonAppleAndBanana/rewards_mean',  # Snake apple banana
        'eval_FrozenLake-Slippery/rewards_mean',         # FL slip
        'eval_FrozenLake-NoSlip/rewards_mean'            # FL non slip
    ]
    
    # Filter to only include environments that exist in our data
    ordered_env_cols = [col for col in desired_order if col in env_cols]
    
    # Create subplots - 2 wide, 3 tall (2x3 grid)
    n_cols = 2
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, metric_col in enumerate(ordered_env_cols):
        ax = axes_flat[i]
        
        # Extract environment name from column name
        env_name_raw = metric_col.replace('eval_', '').replace('/rewards_mean', '')
        # Apply environment name mapping if available, otherwise use raw name
        env_name = eval_env_name_mapping.get(env_name_raw, env_name_raw)
        
        for j, group in enumerate(env_groups):
            # Filter data for this group
            group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
            
            if len(group_data) == 0:
                continue
                
            # Group by step and calculate mean and std
            step_stats = group_data.groupby('_step')[metric_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Only plot if we have data
            if len(step_stats) > 0:
                steps = step_stats['_step']
                means = step_stats['mean']
                stds = step_stats['std']
                
                # Get mapped name and color
                mapped_name = run_name_mapping.get(group, group)
                color = run_name_colors.get(mapped_name, None)
                
                # Ensure we have a valid color - use colorblind palette if not found
                if color is None:
                    # Use colorblind-friendly colors
                    colorblind_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    color = colorblind_colors[j % len(colorblind_colors)]
                
                # Plot mean line
                ax.plot(steps, means, 
                       color=color, 
                       label=f'{mapped_name}',
                       linewidth=2.5)
                
                # Add error bars (std)
                # ax.fill_between(steps, 
                #                means - stds, 
                #                means + stds, 
                #                color=color, 
                #                alpha=0.2)
        
        # Customize subplot
        ax.set_title(f'{env_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Rewards Mean', fontsize=12)
        ax.set_xlim(0, 550)  # Limit x-axis to 550 steps
        ax.grid(True, alpha=0.3)
        ax.legend(**legend_kwargs, fontsize=10)
    
    # Hide empty subplots (we have 6 total subplots in 2x3 grid)
    for i in range(len(ordered_env_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Training Curves: {title_suffix}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    if save_name:
        plt.savefig(os.path.join(figures_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
        print(f"Saved plot as {os.path.join(figures_dir, f'{save_name}.png')}")
    
    plt.show()

# Create separate charts for each environment
print("\n" + "="*60)
print("CREATING FROZENLAKE TRAINED RUNS - ALL EVALUATIONS")
print("="*60)
# create_training_curves('FrozenLake', 'FrozenLake Trained Runs - All Evaluation Environments', 'frozenlake_trained_all_evals')

print("\n" + "="*60)
print("CREATING FASTSNAKE TRAINED RUNS - ALL EVALUATIONS")
print("="*60)
# create_training_curves('FastSnake', 'FastSnake Trained Runs - All Evaluation Environments', 'fastsnake_trained_all_evals')

# ------------------
# Create validation-only training curves
# ------------------

def create_validation_training_curves():
    """
    Create training curves showing each training environment evaluated only on its own validation environment:
    - Snake trained models evaluated on Snake-8Step (FastSnake-Default)
    - FrozenLake trained models evaluated on FrozenLake-NoSlip
    """
    print("\n" + "="*60)
    print("CREATING VALIDATION-ONLY TRAINING CURVES")
    print("="*60)
    
    # Define validation environments for each training environment
    validation_configs = {
        'Snake': {
            'env_groups': [g for g in groups if g.startswith('FS_')],
            'validation_metric': 'eval_FastSnake-Default/rewards_mean',
            'title': 'Snake Trained Models',
            'save_name': 'snake_validation_curves'
        },
        'FrozenLake': {
            'env_groups': [g for g in groups if g.startswith('FL_')],
            'validation_metric': 'eval_FrozenLake-NoSlip/rewards_mean',
            'title': 'FrozenLake Trained Models',
            'save_name': 'frozenlake_validation_curves'
        }
    }
    
    for env_name, config in validation_configs.items():
        print(f"\nCreating {env_name} validation curves...")
        
        # Check if the validation metric exists
        if config['validation_metric'] not in run_eval_metrics_clean.columns:
            print(f"Warning: {config['validation_metric']} not found in data, skipping {env_name}")
            continue
        
        # Create a single subplot for this validation environment
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        
        for j, group in enumerate(config['env_groups']):
            # Filter data for this group
            group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
            
            if len(group_data) == 0:
                continue
                
            # Group by step and calculate mean and std
            step_stats = group_data.groupby('_step')[config['validation_metric']].agg(['mean', 'std', 'count']).reset_index()
            
            # Only plot if we have data
            if len(step_stats) > 0:
                steps = step_stats['_step']
                means = step_stats['mean']
                stds = step_stats['std']
                
                # Get mapped name and color
                original_mapped_name = run_name_mapping.get(group, group)
                mapped_name = original_mapped_name
                
                # Remove "Snake" or "FrozenLake" from the beginning of the name
                if mapped_name.startswith('Snake '):
                    mapped_name = mapped_name[6:]  # Remove "Snake "
                elif mapped_name.startswith('FrozenLake '):
                    mapped_name = mapped_name[11:]  # Remove "FrozenLake "
                
                color = run_name_colors.get(original_mapped_name, None)
                
                # Ensure we have a valid color - use colorblind palette if not found
                if color is None:
                    # Use colorblind-friendly colors
                    colorblind_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    color = colorblind_colors[j % len(colorblind_colors)]
                
                # Plot mean line
                ax.plot(steps, means, 
                       color=color, 
                       label=f'{mapped_name}',
                       linewidth=2.5)
                
                # Add error bars (std) - commented out as per user preference
                # ax.fill_between(steps, 
                #                means - stds, 
                #                means + stds, 
                #                color=color, 
                #                alpha=0.2)
        
        # Customize the plot
        # ax.set_title(f'{config["title"]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=15)
        ax.set_ylabel('Rewards Mean', fontsize=15)
        ax.set_xlim(0, 550)  # Limit x-axis to 550 steps
        ax.grid(True, alpha=0.3)
        ax.legend(**legend_kwargs, fontsize=11)
        
        plt.tight_layout()
        
        # Save the plot
        if config['save_name']:
            save_path = os.path.join(figures_dir, f'{config["save_name"]}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved validation plot as {save_path}")
        
        plt.show()

# Create validation-only training curves
create_validation_training_curves()

# print("\n" + "="*60)
# print("CREATING ALL TRAINING ENVIRONMENTS - ALL EVALUATIONS")
# print("="*60)
# create_training_curves('All', 'All Training Environments - All Evaluation Environments', 'all_training_all_evals')

# ------------------
# Create Before/After Comparison Table
# ------------------

def create_before_after_table():
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
            # For paired t-test, we need to match experiments
            # Get experiment names for both steps
            step_before_experiments = group_data[group_data['_step'] == step_before]['experiment_name'].tolist()
            step_after_experiments = group_data[group_data['_step'] == step_after]['experiment_name'].tolist()
            
            # Find common experiments
            common_experiments = set(step_before_experiments) & set(step_after_experiments)
            
            if len(common_experiments) >= 2:  # Need at least 2 pairs for t-test
                # Get paired data
                paired_before = []
                paired_after = []
                
                for exp in common_experiments:
                    before_val = group_data[(group_data['_step'] == step_before) & (group_data['experiment_name'] == exp)][metric].iloc[0]
                    after_val = group_data[(group_data['_step'] == step_after) & (group_data['experiment_name'] == exp)][metric].iloc[0]
                    
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
                env_name_raw = metric.replace('eval_', '').replace('/rewards_mean', '')
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
    
    # Save to CSV
    output_file = 'before_after_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\nTable saved to: {output_file}")
    
    # Save statistical results
    if statistical_results:
        stats_output_file = 'statistical_results.csv'
        stats_df.to_csv(stats_output_file, index=False)
        print(f"Statistical results saved to: {stats_output_file}")
    
    return comparison_df, statistical_results

# Create the before/after comparison table
before_after_table, statistical_results = create_before_after_table()

# Export to LaTeX
def export_to_latex(comparison_df, filename='before_after_comparison.tex'):
    """
    Export the comparison table to LaTeX format with proper formatting.
    Creates two separate tables: one for Snake tasks and one for FrozenLake tasks.
    Each table has separate columns for Algorithm (PPO/GRPO) and Model Size.
    """
    print(f"\nExporting tables to LaTeX: {filename}")
    
    # Define metrics for each environment
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
    
    def create_table_data(metrics, env_name):
        """Helper function to create table data for a specific environment"""
        table_data = []
        
        # Group by algorithm first (PPO, then GRPO)
        ppo_groups = [g for g in target_groups if 'PPO' in g]
        grpo_groups = [g for g in target_groups if 'GRPO' in g]
        
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
                env_name_raw = metric.replace('eval_', '').replace('/rewards_mean', '')
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
                
                # Add row to table
                table_data.append({
                    'Algorithm': algorithm_name,
                    'Model Size': model_size,
                    'Task': display_name,
                    'Step 0': step_0_formatted,
                    'Step 550': step_550_formatted,
                    '$\\Delta$': delta_formatted
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
                env_name_raw = metric.replace('eval_', '').replace('/rewards_mean', '')
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
                
                # Add row to table
                table_data.append({
                    'Algorithm': algorithm_name,
                    'Model Size': model_size,
                    'Task': display_name,
                    'Step 0': step_0_formatted,
                    'Step 550': step_550_formatted,
                    '$\\Delta$': delta_formatted
                })
        
        return table_data
    
    # Create data for both tables
    snake_data = create_table_data(snake_metrics, 'Snake')
    frozenlake_data = create_table_data(frozenlake_metrics, 'FrozenLake')
    
    # Create DataFrames
    snake_df = pd.DataFrame(snake_data)
    frozenlake_df = pd.DataFrame(frozenlake_data)
    
    # Generate LaTeX tables
    snake_latex = snake_df.to_latex(
        index=False,
        escape=False,
        caption="Snake Environment Performance: Initial vs Final",
        label="tab:snake_comparison",
        column_format='l' + 'l' + 'l' + 'c' * 3,  # l for text columns, c for numeric columns
        position='htbp'
    )
    
    frozenlake_latex = frozenlake_df.to_latex(
            index=False,
            escape=False,
        caption="FrozenLake Environment Performance: Initial vs Final",
        label="tab:frozenlake_comparison",
        column_format='l' + 'l' + 'l' + 'c' * 3,  # l for text columns, c for numeric columns
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
latex_output = export_to_latex(before_after_table)
