#!/bin/bash
# Local Evaluation Wrapper Script
# 
# This script provides a convenient way to run evaluations on different models
# and configurations locally without using SLURM sbatch.
#
# Usage:
#   ./eval_local.sh <model_type> <model_path> [eval_config] [run_name] [gpus] [ppo_config_override]
#
# Examples:
#   ./eval_local.sh checkpoint /path/to/checkpoint eval_1 "my_eval_run"
#   ./eval_local.sh base_model Qwen/Qwen2.5-3B-Instruct eval_1 "base_model_eval"
#   ./eval_local.sh checkpoint /path/to/checkpoint eval_1 "my_run" 1
#   ./eval_local.sh checkpoint /path/to/checkpoint eval_1 "" 1 "evaluation/eval_config_sampling.yaml"
#   ./eval_local.sh checkpoint /path/to/checkpoint  # Uses default eval_1 config

# ./Verlog/evaluation/eval_local.sh base_model Qwen/Qwen2.5-3B-Instruct eval_1_no_babyAI eval_local_qwen3B 2 Verlog/evaluation/eval_config_sampling.yaml

set -e

# Default values
DEFAULT_EVAL_CONFIG="eval_1"
DEFAULT_WANDB_PROJECT="AAMAS_msrl_base"
DEFAULT_OUTPUT_DIR="/scratch/jsbd1n24/evaluation_results"
DEFAULT_GPUS=1
DEFAULT_PPO_CONFIG_OVERRIDE="Verlog/evaluation/eval_config.yaml"

# Function to print usage
print_usage() {
    echo "Usage: $0 <model_type> <model_path> [eval_config] [run_name] [gpus] [ppo_config_override]"
    echo ""
    echo "Arguments:"
    echo "  model_type      : 'checkpoint' or 'base_model'"
    echo "  model_path      : Path to checkpoint directory or base model name/path"
    echo "  eval_config     : Evaluation configuration name (default: eval_1)"
    echo "  run_name        : Custom run name for wandb (use \"\" for auto-generated)"
    echo "  gpus            : Number of GPUs to use (default: 1)"
    echo "  ppo_config_override: Path to evaluation config YAML file (default: Verlog/evaluation/eval_config.yaml)"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoint /scratch/jsbd1n24/checkpoints/AAMAS_msrl/FS_PPO_3B_run5/global_step_600/actor"
    echo "  $0 base_model Qwen/Qwen2.5-3B-Instruct eval_1 \"base_model_eval\""
    echo "  $0 checkpoint /path/to/checkpoint eval_1 \"my_run\" 1"
    echo "  $0 checkpoint /path/to/checkpoint eval_1 \"\" 1 \"Verlog/evaluation/eval_config_sampling.yaml\""
    echo "  $0 base_model Qwen/Qwen2.5-3B-Instruct eval_1 \"\" 1 \"Verlog/evaluation/eval_config_deterministic.yaml\""
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
PPO_CONFIG_OVERRIDE="${6:-$DEFAULT_PPO_CONFIG_OVERRIDE}"

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
echo "VERL Local Evaluation Runner"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Model path: $MODEL_PATH"
echo "Evaluation config: $EVAL_CONFIG"
echo "Run name: $RUN_NAME"
echo "Wandb project: $DEFAULT_WANDB_PROJECT"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Eval config file: $PPO_CONFIG_OVERRIDE"
echo "=========================================="

# Check if model path exists (for checkpoints)
if [[ "$MODEL_TYPE" == "checkpoint" && ! -d "$MODEL_PATH" ]]; then
    echo "Warning: Checkpoint directory does not exist: $MODEL_PATH"
    echo "Continuing anyway (might be a remote path)..."
fi

# Check if evaluation config exists
PPO_CONFIG_OVERRIDE_CHECK="Verlog/verl/trainer/config/evaluation/${EVAL_CONFIG}.yaml"
if [[ ! -f "$PPO_CONFIG_OVERRIDE_CHECK" ]]; then
    echo "Error: Evaluation config file not found: $PPO_CONFIG_OVERRIDE_CHECK"
    echo "Available configs:"
    ls -1 Verlog/verl/trainer/config/evaluation/*.yaml | sed 's/.*\//  - /' | sed 's/\.yaml$//'
    exit 1
fi

# Create logs directory
mkdir -p logs

# --- Setup Environment ---
echo "Setting up environment..."

# Load modules (if available on the system)
if command -v module >/dev/null 2>&1; then
    echo "Loading modules..."
    module load conda/python3 2>/dev/null || echo "Warning: Could not load conda/python3 module"
    module load cuda/12.4.0 2>/dev/null || echo "Warning: Could not load cuda/12.4.0 module"
    module load gcc/13.2.0 2>/dev/null || echo "Warning: Could not load gcc/13.2.0 module"
else
    echo "Module system not available, using system Python and CUDA"
fi

# Change to the correct directory (get the directory where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
echo "Changed to project root: $(pwd)"

# Activate conda environment (if available)
if command -v conda >/dev/null 2>&1; then
    echo "Activating conda environment..."
    source activate verlog_new 2>/dev/null || echo "Warning: Could not activate verlog_new conda environment"
else
    echo "Conda not available, using system Python"
fi

# Set environment variables
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# --- Configure Environment for Ray ---
export RAY_OBJECT_STORE_MEMORY=50000000000

# Additional Ray configuration for stability
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0

# Set CUDA environment variables
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

echo "Environment setup complete!"

# --- Run the evaluation ---
echo "Starting local evaluation..."
echo "Model path: $MODEL_PATH"
echo "Evaluation config: $EVAL_CONFIG"
echo "Wandb project: $DEFAULT_WANDB_PROJECT"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Eval config file: $PPO_CONFIG_OVERRIDE"

# Create log file name
LOG_FILE="logs/eval_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Run the evaluation using Python directly
echo "Running evaluation and logging to: $LOG_FILE"
python3 Verlog/evaluation/run_evaluation.py \
  --model_path "$MODEL_PATH" \
  --eval_config "$EVAL_CONFIG" \
  --config_path "Verlog/verl/trainer/config/ppo_trainer.yaml" \
  --wandb_project "$DEFAULT_WANDB_PROJECT" \
  --wandb_run_name "$RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --gpus "$GPUS" \
  --nodes 1 \
  --ppo_config_override "$PPO_CONFIG_OVERRIDE" 2>&1 | tee "$LOG_FILE"

# Check if the evaluation was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "EVALUATION COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo "Wandb run: $DEFAULT_WANDB_PROJECT/$RUN_NAME"
    echo ""
    echo "To view the results:"
    echo "  - Check the output directory: $OUTPUT_DIR"
    echo "  - View the log file: $LOG_FILE"
    echo "  - Check wandb dashboard for metrics"
else
    echo ""
    echo "=========================================="
    echo "EVALUATION FAILED!"
    echo "=========================================="
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi
