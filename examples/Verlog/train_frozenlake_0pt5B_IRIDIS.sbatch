#!/bin/bash
#SBATCH --job-name=verlog-PPO-FrozenLake-0pt5
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=150GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=l40
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1

# --- Setup ---
module load conda/python3
module load cuda/12.4.0
module load gcc/13.2.0

cd Verlog
source activate verlog

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

# --- Define model and batch size ---
model=Qwen/Qwen2.5-0.5B-Instruct
micro_batch_size=8

# --- Run your main application ---
PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  critic.model.path=$model \
  critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  critic.highlight_first=True \
  critic.highlight_ratio=3.0 \
  algorithm.step_gamma=0.99 \
  algorithm.step_lam=0.95 \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_ctrl.kl_coef=0.1 \
  envs.n_rollouts=32 \
  envs.duplication_mode=none \
  envs.freeze_completed_episodes=True \
  envs.env_name=frozenlake \
  envs.format_penalty=1 \
  envs.binary_reward=True \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.frozenlake_kwargs.random_map_size=4 \
  envs.frozenlake_kwargs.frozen_probability=0.80 \
  envs.frozenlake_kwargs.is_slippery=false \
  envs.frozenlake_kwargs.print_visualization=true \
  envs.frozenlake_kwargs.print_coordinates=true \
  envs.frozenlake_kwargs.max_rounds=8 \
  trainer.project_name=AAMAS_msrl \
  trainer.experiment_name=frozenlake_0pt5B_T1pt5_k3 \
  trainer.val_before_train=True \
  trainer.critic_warmup=2 \
  trainer.critic_warmup_step=2 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=1000000 \
  trainer.test_freq=30 \
  trainer.render=False \
  trainer.total_epochs=500 \
  evaluation=eval_1 2>&1 | tee verl_frozenlake_demo.log

# --- Cleanup Ray Cluster ---
if [ -n "$RAY_HEAD_PID" ]; then
    echo "Stopping Ray cluster (PID: $RAY_HEAD_PID)..."
    ray stop
    kill $RAY_HEAD_PID 2>/dev/null || true
    echo "Ray cluster stopped."
fi
