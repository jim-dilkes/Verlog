import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from analysis_configs import legend_kwargs, eval_colors_list

# Set Seaborn style for better looking plots
sns.set_theme(style="whitegrid")
# Don't set a global palette as we'll control colors explicitly
plt.style.use('default')
# Import configurations from default_configs.py
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',  # More formal for academic papers
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


# Files downloaded fr
# om wandb
dqn_file = 'analysis/data/dqn_eval_at_frame.csv'
ppo_file = 'analysis/data/ppo_eval_at_step_2.csv'

dqn_df = pd.read_csv(dqn_file)
ppo_df = pd.read_csv(ppo_file)

# For DQN: Average the two experiment runs
dqn_df['Mean Environment Score'] = dqn_df[['dqn-exp_2-361344-20251021-165948 - eval_avg_score', 
                                              'dqn-exp_2-361336-20251021-165716 - eval_avg_score']].mean(axis=1)
dqn_df['Episodes'] = dqn_df['Step']

# For PPO: Extract 14B and 7B scores and multiply steps by 8 * 32 = 256
ppo_14b_df = pd.DataFrame()
ppo_14b_df['Episodes'] = ppo_df['Step'] * 7 * 32
ppo_14b_df['Mean Environment Score'] = ppo_df['Group: FS_PPO_14B - eval_Snake-20Step/score_mean']

ppo_7b_df = pd.DataFrame()
ppo_7b_df['Episodes'] = ppo_df['Step'] * 7 * 32
ppo_7b_df['Mean Environment Score'] = ppo_df['Group: FS_PPO_7B - eval_Snake-20Step/score_mean']

# Keep only relevant DQN columns
dqn_df = dqn_df[['Episodes', 'Mean Environment Score']]

print("DQN data:")
print(dqn_df.head())
print("\nPPO 14B data:")
print(ppo_14b_df.head())
print("\nPPO 7B data:")
print(ppo_7b_df.head())

# Filter all dataframes to episodes before x_limit
x_limit = 500_000
dqn_df = dqn_df[dqn_df['Episodes'] <= x_limit]
ppo_14b_df = ppo_14b_df[ppo_14b_df['Episodes'] <= x_limit]
ppo_7b_df = ppo_7b_df[ppo_7b_df['Episodes'] <= x_limit]

# Remove NaN values from PPO dataframes
ppo_14b_df = ppo_14b_df.dropna()
ppo_7b_df = ppo_7b_df.dropna()

# Create a second plot specifically for Score
# Define colors based on model sizes
size_colors = {
    '7B': '#029E73',   # Green
    '14B': '#DE8F05',  # Yellow/Orange
    'DQN': '#000000'   # Black
}

plt.figure(figsize=(6, 4))
plt.plot(dqn_df['Episodes'], dqn_df['Mean Environment Score'], label='DQN', color=size_colors['DQN'], alpha=0.8)
plt.plot(ppo_14b_df['Episodes'], ppo_14b_df['Mean Environment Score'], label='PPO 14B', color=size_colors['14B'], alpha=0.8)
plt.plot(ppo_7b_df['Episodes'], ppo_7b_df['Mean Environment Score'], label='PPO 7B', color=size_colors['7B'], alpha=0.8)

plt.xticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000], ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500'])

plt.xlabel('Frame (Thousands)')
plt.ylabel('Mean Score')
plt.legend(loc='lower right')
plt.savefig('analysis/charts/sample_complexity_score_dqn_ppo.png', dpi=600)
plt.show()