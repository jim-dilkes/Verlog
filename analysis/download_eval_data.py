import os
import wandb
import pandas as pd
import json
from collections import defaultdict
import pathlib

output_dir = pathlib.Path(__file__).parent / "data"

experiment_groups = [
    "FS_PPO_7B",
    "FS_PPO_3B",
    "FS_PPO_0pt5B",
    "FS_GRPO_3B",
    "FS_GRPO_0pt5B",
    "FL_PPO_3B",
    "FL_PPO_0pt5B",
    "FL_GRPO_3B",
    "FL_GRPO_0pt5B",
    ]
download_groups = experiment_groups

api = wandb.Api()

entity, project = "jimdilkes", "AAMAS_msrl"
# Filter runs by group
runs = [run for run in api.runs(entity + "/" + project) if run.group in download_groups or run.group in experiment_groups]

print(f"Found {len(runs)} runs in project {project} with groups {download_groups}")

# Print information about each run
for run in runs:
    print(f"Run ID: {run.id}, Group: {run.group}, Created: {run.created_at}")

# Scan all runs and gather ALL keys, deduplicated
all_run_keys = set()
for run in runs:
    # Safely retrieve the summary as a dictionary,
    # defaulting to an empty dict {} if it is malformed or missing.
    summary_dict = getattr(run.summary, "_json_dict", {})
    
    # If the summary object is somehow an empty string directly, 
    # the above line might still fail or return a string. 
    # Let's ensure it's a dictionary before calling keys().
    if not isinstance(summary_dict, dict):
        summary_dict = {}

    all_run_keys.update(list(summary_dict.keys()))

# # Get the last run's keys for specific metrics
# last_run_keys = list(runs[-1].summary.keys())

print("Printing all eval keys:")
for k in all_run_keys:
    if "eval_" in k:
        print(k)


# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Extract eval reward mean keys
eval_reward_mean_keys = [k for k in all_run_keys if k.startswith("eval_") and k.endswith("/reward_mean")]
eval_success_mean_keys = [k for k in all_run_keys if k.startswith("eval_") and k.endswith("/success_rate")]
train_episode_reward_mean_keys = [k for k in all_run_keys if "reward" in k and k.endswith("/episode_mean")]
train_step_reward_mean_keys = [k for k in all_run_keys if "reward" in k and k.endswith("/step_mean")]
train_episode_success_rate_keys = ["generation/success_rate"]


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary.
    
    Now safely handles inputs that are a JSON-formatted string representing a dict,
    coercing them into a dictionary before flattening.
    """
    
    # === NEW: Handle JSON string input ===
    if isinstance(d, str):
        try:
            # Attempt to parse the string as a dictionary
            d = json.loads(d)
        except json.JSONDecodeError:
            # If it fails to parse (e.g., it's a simple string like 'N/A'), 
            # treat it as an empty dictionary to prevent crashing.
            return {}
        except TypeError:
            # Handle cases where the string might be NoneType or similar, 
            # though isinstance(d, str) should prevent this.
            return {}
    
    # If the input isn't a dictionary after the string check, return empty
    if not isinstance(d, dict):
        return {}
    # === END NEW: Handle JSON string input ===

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # Recursive call for nested dictionaries
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)

# Process run histories
run_histories = []
for run in runs:
    # Generating keys
    # 1. Attempt to get the underlying summary data
    summary_data = getattr(run.summary, "_json_dict", run.summary)
    # 2. Check and coerce the data to a dictionary
    if isinstance(summary_data, dict):
        # Case 1: Data is already a proper dictionary (Success)
        summary_dict = summary_data
    elif isinstance(summary_data, str) and summary_data.strip().startswith('{'):
        # Case 2: Data is a JSON string (The fix for your issue)
        try:
            # Attempt to parse the string into a dictionary
            summary_dict = json.loads(summary_data)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON string for run {run.id}. Skipping.")
            summary_dict = {}
    else:
        # Case 3: Data is None, an empty string, or some other malformed object
        summary_dict = {}
    this_run_keys = list(summary_dict.keys())
    eval_reward_mean_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/rewards_mean")]
    eval_score_mean_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/score_mean")]
    eval_tokens_per_rollout_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/tokens_per_rollout")]
    eval_tokens_per_step_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/tokens_per_step")]
    eval_traj_length_mean_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/traj_length_mean")]
    eval_pos_reward_any_prop_mean_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/pos_reward_any_prop_mean")]
    eval_inference_time_seconds_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/inference_time_seconds")]
    eval_inference_time_per_rollout_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/inference_time_per_rollout")]
    eval_inference_time_per_step_keys = [k for k in this_run_keys if k.startswith("eval_") and k.endswith("/inference_time_per_step")]

    train_reward_mean_keys = [k for k in this_run_keys if k.startswith("val/") and k.endswith("/rewards_mean")]
    train_traj_length_mean_keys = [k for k in all_run_keys if k.startswith("val/") and k.endswith("/traj_length_mean")]

    all_keys = eval_reward_mean_keys + eval_score_mean_keys + eval_tokens_per_rollout_keys + eval_tokens_per_step_keys + eval_traj_length_mean_keys + eval_pos_reward_any_prop_mean_keys + eval_inference_time_seconds_keys + eval_inference_time_per_rollout_keys + eval_inference_time_per_step_keys + train_reward_mean_keys + train_traj_length_mean_keys

    print(f"Processing history for run {run.id} (group: {run.group})")
    try:
        run_history = run.history(keys=all_keys)
        print(f"Successfully loaded history with {len(run_history)} rows")
        run_history["run_id"] = run.id
        run_histories.append(run_history)
    except Exception as e:
        print(f"Error loading history for run {run.id}: {str(e)}")

if len(run_histories) > 0:
    run_histories_df = pd.concat(run_histories)
    eval_rewards_filename = os.path.join(output_dir, "eval_metrics.csv")
    run_histories_df.to_csv(eval_rewards_filename, index=False)
    print(f"\nSaved histories for {len(run_histories)} runs to {eval_rewards_filename}")
else:
    print("\nWARNING: No run histories were successfully loaded!")

# Process run configurations
run_configs = []
run_groups = []
for run in runs:
    # Flatten the nested config dictionary
    flat_config = flatten_dict(run.config)

        # === START ROBUST FILE DOWNLOAD ===
    meta = {}
    try:
        # Download file to a temporary location (download(replace=True) does this)
        metadata_file = run.file("wandb-metadata.json").download(replace=True)
        # Load the content
        with open(metadata_file.name, "r") as f:
            meta = json.load(f)
        # Clean up the downloaded file
        os.remove(metadata_file.name)
    except Exception as e:
        print(f"Warning: Could not download or process wandb-metadata.json for run {run.id}: {e}")
        # Use default empty values if download fails
        meta = {'host': 'N/A', 'gpu': 'N/A'} 
    # === END ROBUST FILE DOWNLOAD ===

    run_groups.append(run.group)

    # Add run_id to the flattened config
    flat_config['run_id'] = run.id
    flat_config['hostname'] = meta['host']
    flat_config['gpu_type'] = meta['gpu_type']
    # Add group and tags information
    flat_config['group'] = run.group if run.group else ''
    flat_config['tags'] = ','.join(run.tags) if run.tags else ''
    run_configs.append(flat_config)

# Convert to DataFrame
run_configs_df = pd.DataFrame(run_configs)
print(f"Run groups: {set(run_groups)}")
# Save to CSV
run_configs_filename = os.path.join(output_dir, "run_configs.csv")
run_configs_df.to_csv(run_configs_filename, index=False)

print(f"Saved {len(run_configs)} run configurations to {run_configs_filename}")

