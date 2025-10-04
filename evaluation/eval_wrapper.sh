#!/bin/bash
# Evaluation Wrapper Script
# 
# This script provides a convenient way to run evaluations on different models
# and configurations without having to remember all the command-line arguments.
#
# Usage:
#   ./eval_wrapper.sh <model_type> <model_path> [eval_config] [run_name] [gpus] [partition] [time] [ppo_config_override]
#
# Examples:
#   ./eval_wrapper.sh checkpoint /path/to/checkpoint eval_1 "my_eval_run"
#   ./eval_wrapper.sh base_model Qwen/Qwen2.5-3B-Instruct eval_1 "base_model_eval"
#   ./eval_wrapper.sh checkpoint /path/to/checkpoint eval_1 "my_run" 2 "quad_h200" "04:00:00"
#   ./eval_wrapper.sh checkpoint /path/to/checkpoint eval_1 "" 1 "gpu" "01:00:00" "evaluation/eval_config_sampling.yaml"
#   ./eval_wrapper.sh checkpoint /path/to/checkpoint  # Uses default eval_1 config

set -e

# Default values
DEFAULT_EVAL_CONFIG="eval_1"
DEFAULT_WANDB_PROJECT="AAMAS_msrl_base"
DEFAULT_OUTPUT_DIR="/scratch/jsbd1n24/evaluation_results"
DEFAULT_GPUS=1
DEFAULT_PARTITION="quad_h200"
DEFAULT_TIME="02:00:00"
DEFAULT_PPO_CONFIG_OVERRIDE="Verlog/evaluation/eval_config.yaml"

# Function to print usage
print_usage() {
    echo "Usage: $0 <model_type> <model_path> [eval_config] [run_name] [gpus] [partition] [time] [ppo_config_override]"
    echo ""
    echo "Arguments:"
    echo "  model_type      : 'checkpoint' or 'base_model'"
    echo "  model_path      : Path to checkpoint directory or base model name/path"
    echo "  eval_config     : Evaluation configuration name (default: eval_1)"
    echo "  run_name        : Custom run name for wandb (use \"\" for auto-generated)"
    echo "  gpus            : Number of GPUs to use (default: 1)"
    echo "  partition       : SLURM partition to use (default: quad_h200)"
    echo "  time            : Maximum runtime in HH:MM:SS format (default: 02:00:00)"
    echo "  ppo_config_override: Path to evaluation config YAML file (default: evaluation/eval_config.yaml)"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoint /scratch/jsbd1n24/checkpoints/AAMAS_msrl/FS_PPO_3B_run5/global_step_600/actor"
    echo "  $0 base_model Qwen/Qwen2.5-3B-Instruct eval_1 \"base_model_eval\""
    echo "  $0 checkpoint /path/to/checkpoint eval_1 \"my_run\" 2"
    echo "  $0 checkpoint /path/to/checkpoint eval_1 \"\" 1 \"gpu\" \"01:00:00\" \"evaluation/eval_config_sampling.yaml\""
    echo "  $0 base_model Qwen/Qwen2.5-3B-Instruct eval_1 \"\" 1 \"gpu\" \"01:00:00\" \"evaluation/eval_config_deterministic.yaml\""
    echo ""
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Check minimum arguments
if [[ $# -lt 2 ]]; then
    echo "Error: Missing required arguments"
    echo ""
    print_usage
    exit 1
fi

# Parse arguments
MODEL_TYPE="$1"
MODEL_PATH="$2"
EVAL_CONFIG="${3:-$DEFAULT_EVAL_CONFIG}"
RUN_NAME="${4:-}"
GPUS="${5:-$DEFAULT_GPUS}"
PARTITION="${6:-$DEFAULT_PARTITION}"
TIME="${7:-$DEFAULT_TIME}"
PPO_CONFIG_OVERRIDE="${8:-$DEFAULT_PPO_CONFIG_OVERRIDE}"

# Validate model type
if [[ "$MODEL_TYPE" != "checkpoint" && "$MODEL_TYPE" != "base_model" ]]; then
    echo "Error: model_type must be 'checkpoint' or 'base_model'"
    echo "Got: $MODEL_TYPE"
    exit 1
fi

# Generate run name if not provided
if [[ -z "$RUN_NAME" ]]; then
    if [[ "$MODEL_TYPE" == "checkpoint" ]]; then
        # Extract model name from checkpoint path
        MODEL_NAME=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        RUN_NAME="eval_${MODEL_NAME}_${TIMESTAMP}"
    else
        # Use base model name
        MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/[^a-zA-Z0-9_-]/_/g')
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        RUN_NAME="eval_${MODEL_NAME}_${TIMESTAMP}"
    fi
fi

# Create output directory with run name
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}/${RUN_NAME}"

echo "=========================================="
echo "VERL Evaluation Runner"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Model path: $MODEL_PATH"
echo "Evaluation config: $EVAL_CONFIG"
echo "Run name: $RUN_NAME"
echo "Wandb project: $DEFAULT_WANDB_PROJECT"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Partition: $PARTITION"
echo "Runtime: $TIME"
echo "Eval config file: $PPO_CONFIG_OVERRIDE"
echo "=========================================="

# Check if model path exists (for checkpoints)
if [[ "$MODEL_TYPE" == "checkpoint" && ! -d "$MODEL_PATH" ]]; then
    echo "Warning: Checkpoint directory does not exist: $MODEL_PATH"
    echo "Continuing anyway (might be a remote path)..."
fi

# Check if evaluation config exists
if [[ ! -f "$PPO_CONFIG_OVERRIDE" ]]; then
    echo "Error: Evaluation config file not found: $PPO_CONFIG_OVERRIDE"
    echo "Available configs:"
    ls -1 Verlog/verl/trainer/config/evaluation/*.yaml | sed 's/.*\//  - /' | sed 's/\.yaml$//'
    exit 1
fi

# Create logs directory
mkdir -p logs

# Run the evaluation using sbatch
echo "Submitting evaluation job to SLURM..."
sbatch \
    --job-name="eval_${RUN_NAME}" \
    --output="logs/eval_${RUN_NAME}_%j.out" \
    --error="logs/eval_${RUN_NAME}_%j.err" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --gpus-per-node="$GPUS" \
    Verlog/evaluation/run_evaluation.sbatch \
    "$MODEL_PATH" \
    "$EVAL_CONFIG" \
    "$DEFAULT_WANDB_PROJECT" \
    "$RUN_NAME" \
    "$OUTPUT_DIR" \
    "$GPUS" \
    "$PPO_CONFIG_OVERRIDE"

echo "Evaluation job submitted successfully!"
echo "Monitor progress with: squeue -u \$USER"
echo "View logs in: logs/eval_${RUN_NAME}_*.out"
