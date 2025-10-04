#!/bin/bash
# Example Usage Script for VERL Evaluation Suite
#
# This script demonstrates how to use the evaluation suite with different
# types of models and configurations.

set -e

echo "=========================================="
echo "VERL Evaluation Suite - Example Usage"
echo "=========================================="

# Example 1: Evaluate a base model (before training)
echo ""
echo "Example 1: Evaluating a base model"
echo "-----------------------------------"
echo "Command:"
echo "./eval_wrapper.sh base_model Qwen/Qwen2.5-3B-Instruct eval_1 \"base_model_baseline\""
echo ""
echo "This will:"
echo "- Load the base Qwen model"
echo "- Run evaluation on all environments in eval_1 config"
echo "- Log results to wandb project 'AAMAS_msrl_base'"
echo "- Save results with run name 'base_model_baseline'"

# Example 2: Evaluate a checkpoint (after training)
echo ""
echo "Example 2: Evaluating a trained checkpoint"
echo "------------------------------------------"
echo "Command:"
echo "./eval_wrapper.sh checkpoint /scratch/jsbd1n24/checkpoints/AAMAS_msrl/FS_PPO_3B_run5/global_step_600/actor eval_1 \"ppo_600_steps\""
echo ""
echo "This will:"
echo "- Load the PPO checkpoint from step 600"
echo "- Run evaluation on all environments"
echo "- Compare performance against base model"

# Example 3: Evaluate with custom configuration
echo ""
echo "Example 3: Using custom evaluation config"
echo "-----------------------------------------"
echo "Command:"
echo "./eval_wrapper.sh checkpoint /path/to/checkpoint custom_eval \"custom_run\""
echo ""
echo "This will:"
echo "- Use a custom evaluation configuration"
echo "- Run only specific environments defined in custom_eval.yaml"

# Example 4: Evaluate with multiple GPUs
echo ""
echo "Example 4: Using multiple GPUs for faster evaluation"
echo "----------------------------------------------------"
echo "Command:"
echo "./eval_wrapper.sh checkpoint /path/to/checkpoint eval_1 \"fast_eval\" 2"
echo ""
echo "This will:"
echo "- Use 2 GPUs for parallel evaluation"
echo "- Significantly reduce evaluation time"

# Example 5: Direct Python usage
echo ""
echo "Example 5: Direct Python usage (for debugging)"
echo "----------------------------------------------"
echo "Command:"
echo "python Verlog/evaluation/run_evaluation.py \\"
echo "    --model_path /path/to/model \\"
echo "    --eval_config eval_1 \\"
echo "    --wandb_project AAMAS_msrl_base \\"
echo "    --wandb_run_name \"debug_run\" \\"
echo "    --output_dir ./debug_results \\"
echo "    --gpus 1"
echo ""
echo "This will:"
echo "- Run evaluation directly without SLURM"
echo "- Useful for debugging and development"

echo ""
echo "=========================================="
echo "Available Evaluation Configurations:"
echo "=========================================="
echo "- eval_1: Multi-environment suite (FastSnake, BabyAI, FrozenLake)"
echo "- default: Empty configuration (no evaluation)"

echo ""
echo "=========================================="
echo "Output Locations:"
echo "=========================================="
echo "- Results: /scratch/jsbd1n24/evaluation_results/<run_name>/"
echo "- Logs: logs/eval_<run_name>_*.out"
echo "- Wandb: https://wandb.ai/<username>/AAMAS_msrl_base"

echo ""
echo "=========================================="
echo "Monitoring:"
echo "=========================================="
echo "- Check job status: squeue -u \$USER"
echo "- View logs: tail -f logs/eval_<run_name>_*.out"
echo "- Cancel job: scancel <job_id>"

echo ""
echo "Ready to run evaluations! Choose an example above and modify as needed."
