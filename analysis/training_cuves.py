import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats

from analysis_configs import legend_kwargs, run_name_mapping, run_env_mapping, run_alg_mapping, run_name_colors, eval_env_name_mapping
from run_eval_metrics_clean import run_eval_metrics_clean


current_dir = os.path.dirname(os.path.abspath('__file__'))
data_dir = os.path.join(current_dir, 'analysis/data')
figures_dir = os.path.join(current_dir, 'analysis/charts')
os.makedirs(figures_dir, exist_ok=True) 

run_eval_metrics_clean = run_eval_metrics_clean(data_dir)

groups = run_eval_metrics_clean['group'].unique()

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

    run_eval_metrics_clean_filtered = run_eval_metrics_clean[run_eval_metrics_clean['group'] != 'FS_PPO_7B']
    
    for env_name, config in validation_configs.items():
        print(f"\nCreating {env_name} validation curves...")
        
        # Check if the validation metric exists
        if config['validation_metric'] not in run_eval_metrics_clean_filtered.columns:
            print(f"Warning: {config['validation_metric']} not found in data, skipping {env_name}")
            continue
        
        # Create a single subplot for this validation environment
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        
        for j, group in enumerate(config['env_groups']):
            # Filter data for this group
            group_data = run_eval_metrics_clean_filtered[run_eval_metrics_clean_filtered['group'] == group]
            
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
        
        plt.close()

# Create validation-only training curves
create_validation_training_curves()