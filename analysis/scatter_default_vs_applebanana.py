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


def create_snake_scatter_chart():
    """
    Create a scatter chart comparing model performance on Snake-20Step vs Snake-20Steps-PoisonAppleAndBanana.
    Excludes non-256 0.5B and 3B base models, and FL finetuned models.
    """
    print("\n" + "="*60)
    print("CREATING SNAKE SCATTER CHART")
    print("="*60)
    
    # Load base LLM evaluation data (reuse from bar charts)
    import json
    import os
    
    base_llm_data = []
    base_llm_dir = os.path.join(data_dir, 'base_llm_evals')
    
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
                    parts = subdir.split('_')
                    model_size = parts[0].split('-')[1]  # Extract size (0.5B, 3B, etc.)
                    is_4096 = '4096' in subdir
                    is_cot = 'cot' in subdir
                    
                    # Skip finetuned base models (3BFLPPO and 3BFSPPO)
                    if 'FLPPO' in subdir or 'FSPPO' in subdir:
                        continue
                    
                    # Skip non-256 0.5B and 3B base models
                    if (model_size in ['0.5B', '3B']) and is_4096:
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
                            'n_runs': 1
                        })
                except Exception as e:
                    print(f"Error processing {eval_results_file}: {e}")
    
    # Convert to DataFrame and aggregate
    base_llm_df = pd.DataFrame(base_llm_data)
    if len(base_llm_df) > 0:
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
    
    # Get finetuned model data (final performance at step 550)
    finetuned_data = []
    for group in run_eval_metrics_clean['group'].unique():
        # Skip FL (Frozen Lake) finetuned models
        if 'FL_' in group:
            continue
            
        group_data = run_eval_metrics_clean[run_eval_metrics_clean['group'] == group]
        final_data = group_data[group_data['_step'] == 550]
        
        if len(final_data) == 0:
            continue
            
        # Get mapped group name
        mapped_name = run_name_mapping.get(group, group)
        
        # Extract model size
        if '0pt5B' in group:
            model_size = '0.5B'
        elif '3B' in group:
            model_size = '3B'
        elif '7B' in group:
            model_size = '7B'
        else:
            model_size = 'Unknown'
        
        for env in ['Snake-20Step', 'Snake-20Steps-PoisonAppleAndBanana']:
            metric_col = f'eval_{env}/rewards_mean'
            if metric_col in final_data.columns:
                scores = final_data[metric_col].dropna()
                if len(scores) > 0:
                    finetuned_data.append({
                        'model_name': mapped_name,
                        'group': group,
                        'model_size': model_size,
                        'env': env,
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'n_runs': len(scores)
                    })
    
    finetuned_df = pd.DataFrame(finetuned_data)
    
    # Create scatter plot data
    scatter_data = []
    
    # Add base LLM data
    for _, row in base_llm_df.iterrows():
        if row['env'] in ['Snake-20Step', 'Snake-20Steps-PoisonAppleAndBanana']:
            scatter_data.append({
                'model_name': f"{row['model_size']}B-{'4096' if row['is_4096'] else '256'}{'-CoT' if row['is_cot'] else ''}",
                'model_type': 'base_llm',
                'model_size': row['model_size'],
                'env': row['env'],
                'mean_score': row['mean_score'],
                'std_score': row['std_score']
            })
    
    # Add finetuned model data
    for _, row in finetuned_df.iterrows():
        scatter_data.append({
            'model_name': row['model_name'],
            'model_type': 'finetuned',
            'model_size': row['model_size'],
            'env': row['env'],
            'mean_score': row['mean_score'],
            'std_score': row['std_score']
        })
    
    # Convert to DataFrame and pivot for scatter plot
    scatter_df = pd.DataFrame(scatter_data)
    
    if len(scatter_df) == 0:
        print("No data found for scatter plot!")
        return
    
    # Pivot to get both environments as columns
    pivot_df = scatter_df.pivot_table(
        index=['model_name', 'model_type', 'model_size'], 
        columns='env', 
        values='mean_score', 
        aggfunc='mean'
    ).reset_index()
    
    # Check if we have both required environments
    if 'Snake-20Step' not in pivot_df.columns or 'Snake-20Steps-PoisonAppleAndBanana' not in pivot_df.columns:
        print("Missing required environments for scatter plot!")
        return
    
    # Remove rows with NaN values
    pivot_df = pivot_df.dropna(subset=['Snake-20Step', 'Snake-20Steps-PoisonAppleAndBanana'])
    
    if len(pivot_df) == 0:
        print("No complete data found for scatter plot!")
        return
    
    print(f"Creating scatter plot with {len(pivot_df)} models")
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color scheme (same as bar charts)
    size_colors = {
        '0.5B': '#DE8F05',  # Yellow/Orange
        '3B': '#029E73',    # Green  
        '32B': '#D55E00',   # Orange
        '72B': '#0173B2',   # Blue
        '7B': '#FBAFE4',    # Pink
        'Unknown': '#FBAFE4' # Pink
    }
    
    # Plot base models
    base_models = pivot_df[pivot_df['model_type'] == 'base_llm']
    if len(base_models) > 0:
        for size in base_models['model_size'].unique():
            size_data = base_models[base_models['model_size'] == size]
            ax.scatter(
                size_data['Snake-20Step'], 
                size_data['Snake-20Steps-PoisonAppleAndBanana'],
                c=size_colors.get(size, '#949494'),
                s=100,
                alpha=0.7,
                label=f'{size} Base',
                edgecolors='black',
                linewidth=0.5
            )
    
    # Plot finetuned models with labels
    finetuned_models = pivot_df[pivot_df['model_type'] == 'finetuned']
    if len(finetuned_models) > 0:
        ax.scatter(
            finetuned_models['Snake-20Step'], 
            finetuned_models['Snake-20Steps-PoisonAppleAndBanana'],
            c='#FBAFE4',  # Pink
            s=100,
            alpha=0.7,
            label='Finetuned',
            edgecolors='black',
            linewidth=0.5,
            marker='s'  # Square markers for finetuned
        )
        
        # Add labels for finetuned models
        for _, row in finetuned_models.iterrows():
            ax.annotate(
                row['model_name'], 
                (row['Snake-20Step'], row['Snake-20Steps-PoisonAppleAndBanana']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )
    
    # Add diagonal line (perfect correlation)
    min_val = min(pivot_df['Snake-20Step'].min(), pivot_df['Snake-20Steps-PoisonAppleAndBanana'].min())
    max_val = max(pivot_df['Snake-20Step'].max(), pivot_df['Snake-20Steps-PoisonAppleAndBanana'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
    
    # Customize the plot
    ax.set_xlabel('Snake - 20 Steps', fontsize=12)
    ax.set_ylabel('Snake - Poison Apple, Healthy Banana', fontsize=12)
    ax.set_title('Model Performance: Snake Environments Comparison', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(figures_dir, 'scatter_chart_snake_environments.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter chart as {save_path}")
    
    plt.close()  # Close the figure to free memory

# Create the scatter chart
create_snake_scatter_chart()
