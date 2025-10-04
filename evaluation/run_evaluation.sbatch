#!/bin/bash
#SBATCH --job-name=eval_verl
#SBATCH --output=logs/eval_slurm-%j.out
#SBATCH --error=logs/eval_slurm-%j.err
#SBATCH --mem=250GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=quad_h200
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1

# --- Setup ---
module load conda/python3
module load cuda/12.4.0
module load gcc/13.2.0

cd /iridisfs/home/jsbd1n24/verlog
source activate verlog_new

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

# --- Define evaluation variables ---
# Model path - can be either a checkpoint directory or base model path
MODEL_PATH=${1:-"/scratch/jsbd1n24/checkpoints/AAMAS_msrl/FS_PPO_3B_run5/global_step_600/actor"}
EVAL_CONFIG=${2:-"eval_1"}
WANDB_PROJECT=${3:-"AAMAS_msrl_base"}
WANDB_RUN_NAME=${4:-""}
OUTPUT_DIR=${5:-"/scratch/jsbd1n24/evaluation_results"}
GPUS=${6:-1}
PPO_CONFIG_OVERRIDE=${7:-"Verlog/evaluation/eval_config.yaml"}

# --- Create logs directory ---
mkdir -p logs

# --- Run evaluation ---
echo "Starting evaluation..."
echo "Model path: $MODEL_PATH"
echo "Evaluation config: $EVAL_CONFIG"
echo "Wandb project: $WANDB_PROJECT"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "PPO config override file: $PPO_CONFIG_OVERRIDE"

PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 Verlog/evaluation/run_evaluation.py \
  --model_path "$MODEL_PATH" \
  --eval_config "$EVAL_CONFIG" \
  --config_path "Verlog/verl/trainer/config/ppo_trainer.yaml" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --gpus "$GPUS" \
  --nodes 1 \
  --ppo_config_override "$PPO_CONFIG_OVERRIDE" 2>&1 | tee "logs/eval_${EVAL_CONFIG}_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluation completed!"

# --- Cleanup Ray Cluster ---
if [ -n "$RAY_HEAD_PID" ]; then
    echo "Stopping Ray cluster (PID: $RAY_HEAD_PID)..."
    ray stop
    kill $RAY_HEAD_PID 2>/dev/null || true
    echo "Ray cluster stopped."
fi
