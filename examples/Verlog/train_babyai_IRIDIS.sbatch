#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=swarm_h100
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1

# --- Setup ---
module load conda/python3
module load cuda/12.4.0

cd Verlog
source activate verlog

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# --- Clean up previous Ray instances (this is good practice) ---
ray stop --force 2>/dev/null || true
sleep 2
rm -rf /tmp/ray/session_* 2>/dev/null || true

# --- Configure Environment for Ray ---
export RAY_OBJECT_STORE_MEMORY=50000000000

# Let Ray automatically detect the correct address by not setting RAY_ADDRESS.
# The incorrect/unnecessary environment variables have been removed.

# --- Define model and batch size ---
model=Qwen/Qwen2.5-0.5B-Instruct
# model=/scratch/jsbd1n24/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
micro_batch_size=8

# --- Run your main application ---
# REMOVED the redundant python -c "import ray;..." line.
# Your main application will now handle Ray initialization cleanly.
PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
 data.max_prompt_length=1024 \
 data.max_response_length=384 \
 data.train_batch_size=256 \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
 actor_rollout_ref.model.path=$model \
 critic.model.path=$model \
 critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 critic.highlight_first=True \
 critic.highlight_ratio=3.0 \
 algorithm.step_gamma=0.99 \
 algorithm.step_lam=0.95 \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 envs.n_rollouts=32 \
 envs.env_name=babaisai \
 envs.task=env/two_room-maybe_break_stop-goto_win \
 envs.format_penalty=0.1 \
 envs.binary_reward=True \
 envs.captioner.type=cot \
 envs.captioner.max_text_history=1 \
 envs.captioner.max_cot_history=1 \
 trainer.project_name=zero \
 trainer.experiment_name=test \
 trainer.val_before_train=False \
 trainer.critic_warmup=40  # Multiplier for the train_batch_size for the warmup steps \
 trainer.critic_warmup_step=5 \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=1000000 \
 trainer.test_freq=30 \
 trainer.render=False \
 trainer.total_epochs=300 2>&1 | tee verl_demo.log