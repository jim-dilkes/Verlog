import pandas as pd
import numpy as np
import os 

def run_eval_metrics_clean(data_dir):
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
    # Fix misname: experiment_name FS_PPO_3B_run1 should be FL_PPO_3B_run1
    run_eval_metrics.loc[run_eval_metrics['trainer_value_experiment_name'] == 'FS_PPO_3B_run1', 'trainer_value_experiment_name'] = 'FL_PPO_3B_run1'
    run_eval_metrics.loc[run_eval_metrics['trainer_value_experiment_name'] == 'FS_GRPO_3B_run5_slurm340889', 'trainer_value_experiment_name'] = 'FL_GRPO_0pt5B_run5_slurm340889'
    run_eval_metrics.loc[run_eval_metrics['trainer_value_experiment_name'] == 'FS_GRPO_3B_run4_slurm340887', 'trainer_value_experiment_name'] = 'FS_GRPO_0pt5B_run4_slurm340887'
    run_eval_metrics.loc[run_eval_metrics['trainer_value_experiment_name'] == 'FS_GRPO_3B_run5_slurm340884', 'trainer_value_experiment_name'] = 'FL_GRPO_0pt5B_run5_slurm340884'

    # if the experiment name contains 0pt5, append "_run{run_id}" to the trainer_value_experiment_name
    mask = run_eval_metrics['trainer_value_experiment_name'].str.contains('0pt5')
    run_eval_metrics.loc[mask, 'trainer_value_experiment_name'] = (
        run_eval_metrics.loc[mask, 'trainer_value_experiment_name'] + 
        "_run" + 
        run_eval_metrics.loc[mask, 'run_id'].astype(str)
    )

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
    current_number = 100_000_000

    # Get unique run_ids that have experiments needing numbers
    run_ids_needing_numbers = run_eval_metrics_clean[
        run_eval_metrics_clean['experiment_name'].isin(experiments_needing_numbers)
    ]['run_id'].unique()

    # Assign numbers to each run_id
    for run_id in sorted(run_ids_needing_numbers):
        run_id_to_number[run_id] = current_number
        current_number += 1

    # print(f"\nAssigned run numbers to run_ids:")
    # for run_id, number in run_id_to_number.items():
    #     print(f"  - {run_id}: {number}")

    # Apply the run numbers to experiment names
    for run_id, number in run_id_to_number.items():
        # Find rows with this run_id and experiment names that need numbers
        mask = (run_eval_metrics_clean['run_id'] == run_id) & \
            (run_eval_metrics_clean['experiment_name'].isin(experiments_needing_numbers))
        
        # Update the experiment_name by appending the run number
        run_eval_metrics_clean.loc[mask, 'experiment_name'] = \
            run_eval_metrics_clean.loc[mask, 'experiment_name'] + f'_slurm{number}'

    print(f"\nRun number assignment complete.")

    # Create column for experiment name without slurm run id
    run_eval_metrics_clean['group_run_number'] = run_eval_metrics_clean['experiment_name'].str.replace(r'_slurm\d+$', '', regex=True)

    # --- Optional: Verification (You can remove the duplicate_entries section now) ---

    print(f"\nDe-duplication complete.")
    print(f"Final number of unique rows: {len(run_eval_metrics_clean)}")


    # # Identify Look for duplicate entries - i.e. same experiment_name, same _step
    duplicate_entries = run_eval_metrics_clean[run_eval_metrics_clean.duplicated(subset=['experiment_name', '_step'],keep=False)]
    duplicate_entries[['experiment_name', '_step', 'trainer_value_experiment_name', 'group', 'run_id']].sort_values(by=['experiment_name', '_step'])


    check_exp_per_group = run_eval_metrics_clean.groupby('group')['experiment_name'].nunique()
    print(check_exp_per_group)

    # Save to csv
    run_eval_metrics_clean.to_csv(os.path.join(data_dir, 'run_eval_metrics_clean.csv'), index=False)

    return run_eval_metrics_clean