#!/bin/bash

# Local GPU script for FS_GRPO_0pt5B training
# This script runs the same training as the SLURM version but on local GPU

set -e  # Exit on any error

echo "Timestamp: $(date)"


export CUDA_VISIBLE_DEVICES=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# --- Inject Debugging Variables ---
# The Ray worker crash is almost always a low-level CUDA/runtime failure.
# These flags force immediate, verbose error messages.
export VLLM_LOGGING_LEVEL=DEBUG # Force vLLM (used by verl) to print every step
export NCCL_DEBUG=INFO          # Print information about distributed communication initialization (even if world_size=1)
export TORCH_DISTRIBUTED_DEBUG=DETAIL # PyTorch distributed debug
export CUDA_LAUNCH_BLOCKING=1   # Force synchronous CUDA calls to pinpoint the exact line of crash
# ----------------------------------

# --- Setup ---
module load conda/python3
module load cuda/12.4.0
module load gcc/13.2.0

cd /home/jsbd1n24/verlog/Verlog

# Activate conda environment (adjust path if needed)
# If using conda, uncomment the next line and adjust the environment name
source activate verlog
# If using a different environment setup, modify accordingly

# Set offline mode for Hugging Face
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# --- Clean up previous Ray instances (good practice) ---
echo "Cleaning up previous Ray instances..."
ray stop --force 2>/dev/null || true
sleep 5
rm -rf /tmp/ray/session_* 2>/dev/null || true
rm -rf /tmp/ray/raylet_* 2>/dev/null || true
rm -rf /tmp/ray/plasma_* 2>/dev/null || true
sleep 2

# --- Configure Environment for Ray ---
export RAY_OBJECT_STORE_MEMORY=20000000000

# Additional Ray configuration for stability
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0

# Ray resource configuration for local runs
export RAY_ADDRESS=""
export RAY_DISABLE_STRICT_MODE=1
export RAY_RAYLET_CLIENT_NUM_CONNECT_ATTEMPTS=10
export RAY_RAYLET_CLIENT_CONNECT_TIMEOUT_MILLISECONDS=10000

# --- Define model and batch size ---
model=Qwen/Qwen2.5-0.5B-Instruct
# model=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
micro_batch_size=8

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Run your main application ---
echo "Starting training with model: $model"
echo "Output will be logged to: verl_demo.log"


PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=1500 \
  data.max_response_length=256 \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.rollout.temperature=1.25 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.5 \
  algorithm.adv_estimator=grpo \
  critic.model.path=$model \
  critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  critic.highlight_first=True \
  critic.highlight_ratio=3.0 \
  algorithm.step_gamma=0.99 \
  algorithm.step_lam=0.95 \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  envs.n_rollouts=32 \
  envs.group_initial_seed=random \
  envs.group_rollout_size=16 \
  envs.duplication_mode=none \
  envs.freeze_completed_episodes=True \
  envs.env_name=fastsnake \
  envs.format_penalty=1 \
  envs.binary_reward=False \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.fastsnake_kwargs.width=10 \
  envs.fastsnake_kwargs.height=10 \
  envs.fastsnake_kwargs.max_rounds=8 \
  envs.fastsnake_kwargs.num_external_snakes=1 \
  envs.fastsnake_kwargs.num_random_snakes=1 \
  envs.fastsnake_kwargs.death_reward=-1 \
  trainer.project_name=AAMAS_msrl \
  trainer.experiment_name=FS_GRPO_0pt5B \
  trainer.group=FS_GRPO_0pt5B \
  trainer.val_before_train=True \
  trainer.critic_warmup=40 \
  trainer.critic_warmup_step=5 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.render=False \
  trainer.total_epochs=600 \
  evaluation=eval_1 2>&1 | tee verl_demo.log

# --- Cleanup Ray Cluster ---
echo "Cleaning up Ray cluster..."
conda run -n verlog python3 -c "import ray; ray.shutdown() if ray.is_initialized() else None; print('Ray shutdown completed')" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3
rm -rf /tmp/ray/session_* 2>/dev/null || true

echo "Training completed at: $(date)"
echo "Check verl_demo.log for detailed output"