#!/bin/bash
#SBATCH --job-name=verlog-test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=120GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=quad_h200
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=2

# --- Setup ---
module load conda/python3
module load cuda/12.4.0

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
model=Qwen/Qwen2.5-7B-Instruct
# model=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
micro_batch_size=8

# --- Run your main application ---
PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
  data.max_prompt_length=1500 \
  data.max_response_length=200 \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.rollout.temperature=1.25 \
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
  envs.freeze_completed_episodes=False \
  envs.env_name=fastsnake \
  envs.format_penalty=1 \
  envs.binary_reward=True \
  envs.captioner.type=naive \
  envs.captioner.max_text_history=0 \
  envs.fastsnake_kwargs.width=10 \
  envs.fastsnake_kwargs.height=10 \
  envs.fastsnake_kwargs.max_rounds=5 \
  envs.fastsnake_kwargs.num_external_snakes=1 \
  envs.fastsnake_kwargs.num_random_snakes=1 \
  envs.fastsnake_kwargs.death_reward=-3 \
  trainer.project_name=AAMAS_msrl \
  trainer.experiment_name=fastsnake_7B_T1pt25_death-3_freeze \
  trainer.val_before_train=True \
  trainer.critic_warmup=40 \
  trainer.critic_warmup_step=5 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=1000000 \
  trainer.test_freq=30 \
  trainer.render=False \
  trainer.total_epochs=500 \
  evaluation=eval_1 2>&1 | tee verl_demo.log