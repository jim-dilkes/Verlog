# VERL Evaluation Suite

This directory contains scripts for running evaluation suites on pre-existing VERL models or checkpoints without needing to run the full training pipeline.

## Files

- `run_evaluation.py` - Main evaluation script
- `run_evaluation.sbatch` - SLURM batch script for running evaluations
- `eval_wrapper.sh` - Convenient wrapper script for common evaluation tasks
- `eval_1.yaml` - Multi-environment evaluation configuration
- `default.yaml` - Empty evaluation configuration (no evaluation)

## Quick Start

### Using the Wrapper Script (Recommended)

The easiest way to run evaluations is using the wrapper script:

```bash
# Evaluate a checkpoint
./eval_wrapper.sh checkpoint /path/to/checkpoint/directory

# Evaluate a base model
./eval_wrapper.sh base_model Qwen/Qwen2.5-3B-Instruct

# Evaluate with custom run name
./eval_wrapper.sh checkpoint /path/to/checkpoint eval_1 "my_custom_run"

# Evaluate with multiple GPUs
./eval_wrapper.sh checkpoint /path/to/checkpoint eval_1 "my_run" 2
```

### Using SLURM Directly

```bash
# Submit evaluation job to SLURM
sbatch Verlog/evaluation/run_evaluation.sbatch \
    /path/to/model \
    eval_1 \
    AAMAS_msrl_base \
    "my_run_name" \
    ./results \
    1
```

### Using Python Directly

```bash
# Run evaluation directly with Python
python evaluation/run_evaluation.py \
    --model_path /path/to/model \
    --eval_config eval_1 \
    --wandb_project AAMAS_msrl_base \
    --wandb_run_name "my_run" \
    --output_dir ./results \
    --gpus 1 \
    --eval_config_file evaluation/eval_config.yaml
```

### Using Custom Generation Parameters

```bash
# Use deterministic generation
python evaluation/run_evaluation.py \
    --model_path /path/to/model \
    --eval_config eval_1 \
    --eval_config_file evaluation/eval_config_deterministic.yaml

# Use sampling-based generation
python evaluation/run_evaluation.py \
    --model_path /path/to/model \
    --eval_config eval_1 \
    --eval_config_file evaluation/eval_config_sampling.yaml

# Use longer responses
python evaluation/run_evaluation.py \
    --model_path /path/to/model \
    --eval_config eval_1 \
    --eval_config_file evaluation/eval_config_long_responses.yaml

# Save final configuration to file for record-keeping
python evaluation/run_evaluation.py \
    --model_path /path/to/model \
    --eval_config eval_1 \
    --eval_config_file evaluation/eval_config_sampling.yaml \
    --save_config final_config.json
```

## Configuration

### Evaluation Configurations

The evaluation suite supports multiple environment configurations:

- **eval_1**: Multi-environment evaluation suite including:
  - FastSnake environments (different configurations)
  - BabyAI environments (PickupDist, GoToRedBall, etc.)
  - FrozenLake environments (NoSlip, Slippery, BigSlippery)
  - Snake environments with different reward structures

### Generation Parameters Configuration

The evaluation system uses a separate YAML configuration file (`eval_config.yaml`) to control generation parameters:

- **eval_config.yaml**: Default configuration with deterministic generation
- **eval_config_deterministic.yaml**: Deterministic generation for reproducible results
- **eval_config_sampling.yaml**: Sampling-based generation for diversity
- **eval_config_long_responses.yaml**: Configuration for longer responses

#### Key Parameters

```yaml
model:
  generation:
    temperature: 0.0          # Generation temperature (0.0 = deterministic)
    top_k: -1                 # Top-k sampling (-1 = disabled)
    top_p: 1.0                # Top-p sampling (1.0 = disabled)
    do_sample: false          # Whether to use sampling
    max_response_length: 256  # Maximum response length in tokens
    max_prompt_length: 1500   # Maximum prompt length in tokens
```

### Configuration Verification

The evaluation script automatically prints a comprehensive configuration summary before running, showing:

- **Model Configuration**: Model path, tokenizer settings
- **Generation Parameters**: Temperature, sampling settings, response lengths
- **Evaluation Settings**: Logging, epochs, frequencies
- **Training Configuration**: GPUs, nodes, project settings
- **Environment Configuration**: All environments with their settings
- **Configuration Files**: Paths to all config files used

Example output:
```
================================================================================
FINAL EVALUATION CONFIGURATION
================================================================================

📁 MODEL CONFIGURATION:
  Model path: /path/to/model
  Trust remote code: False
  Use fast tokenizer: True

🎯 GENERATION PARAMETERS:
  Temperature: 0.5
  Top-k: -1
  Top-p: 1.0
  Do sample: False
  Max response length: 256
  Max prompt length: 1500

⚙️  EVALUATION SETTINGS:
  Val before train: False
  Test frequency: -1
  Save frequency: -1
  Total epochs: 1
  Log val generations: 1
  Logger backends: ['console', 'wandb']

🌍 ENVIRONMENT CONFIGURATION:
  Number of environments: 8
    1. FastSnake-Default
       - Rollouts: 100
       - Episode length: 8
       - Environment: fastsnake
    ...

================================================================================
CONFIGURATION VERIFICATION COMPLETE
================================================================================
```

You can also save the final configuration to a JSON file for record-keeping:

```bash
python evaluation/run_evaluation.py \
    --model_path /path/to/model \
    --eval_config eval_1 \
    --save_config final_config.json
```

### Model Types

The script supports two types of model inputs:

1. **Checkpoint Directory**: Path to a VERL training checkpoint
   - Should contain an `actor/` subdirectory with model files
   - Example: `/scratch/jsbd1n24/checkpoints/AAMAS_msrl/FS_PPO_3B_run5/global_step_600/actor`

2. **Base Model**: HuggingFace model name or local path
   - Example: `Qwen/Qwen2.5-3B-Instruct`
   - Example: `/path/to/local/model`

## Output

### Results Files

Evaluation results are saved in multiple formats:

1. **JSON Results**: `evaluation_results_<run_name>.json`
   - Complete evaluation metrics
   - Model information
   - Timestamps and duration

2. **Wandb Logs**: Logged to the specified wandb project
   - Project: `AAMAS_msrl_base` (default)
   - Run name: Auto-generated or custom
   - All metrics are logged with environment prefixes

3. **Console Output**: Real-time progress and summary

### Metrics

The evaluation suite collects comprehensive metrics for each environment:

- **Reward Metrics**: Mean, std, success rates
- **Trajectory Metrics**: Length, completion rates
- **Performance Metrics**: Inference time, token generation rates
- **Environment-Specific Metrics**: Task-specific success indicators

Metrics are prefixed by environment name (e.g., `eval_FastSnake-Default/rewards_mean`).

## Examples

### Evaluate a Trained PPO Model

```bash
# Evaluate a PPO checkpoint on all environments
./eval_wrapper.sh checkpoint \
    /scratch/jsbd1n24/checkpoints/AAMAS_msrl/FS_PPO_3B_run5/global_step_600/actor \
    eval_1 \
    "ppo_600_steps"
```

### Evaluate a Base Model

```bash
# Evaluate a base model before training
./eval_wrapper.sh base_model \
    Qwen/Qwen2.5-3B-Instruct \
    eval_1 \
    "base_model_baseline"
```

### Compare Multiple Models

```bash
# Evaluate multiple checkpoints for comparison
./eval_wrapper.sh checkpoint /path/to/checkpoint_100 eval_1 "model_100_steps"
./eval_wrapper.sh checkpoint /path/to/checkpoint_300 eval_1 "model_300_steps"
./eval_wrapper.sh checkpoint /path/to/checkpoint_600 eval_1 "model_600_steps"
```

## Environment Details

### FastSnake Environments
- **FastSnake-Default**: Standard snake game with apples
- **Snake-20Step**: Extended snake game with 20 steps
- **Snake-20Steps-PoisonAppleAndBanana**: Complex reward structure with poison apples and healthy bananas

### BabyAI Environments
- **BabyAI-PickupDist**: Pick up objects at different distances
- **BabyAI-GoToRedBallNoDists**: Navigate to red ball without distractors
- **BabyAI-GoToRedBall**: Navigate to red ball with distractors

### FrozenLake Environments
- **FrozenLake-NoSlip**: Deterministic navigation
- **FrozenLake-Slippery**: Stochastic navigation with ice
- **FrozenLake-BigSlippery**: Larger map with stochastic navigation

## Troubleshooting

### Common Issues

1. **Model Path Not Found**
   - Ensure the checkpoint directory contains an `actor/` subdirectory
   - For base models, ensure the model name is correct

2. **CUDA Out of Memory**
   - Reduce the number of rollouts in the evaluation config
   - Use fewer GPUs or reduce batch sizes

3. **Ray Initialization Issues**
   - Check that Ray is properly installed
   - Ensure sufficient memory is available

4. **Environment Import Errors**
   - Ensure all required environment packages are installed
   - Check that environment configurations are valid

### Debug Mode

To run in debug mode with more verbose output:

```bash
export RAY_DISABLE_IMPORT_WARNING=0
export VLLM_LOGGING_LEVEL=DEBUG
python Verlog/evaluation/run_evaluation.py [args...]
```

## Customization

### Adding New Environments

To add new environments to the evaluation suite:

1. Add environment configuration to `eval_1.yaml`
2. Ensure the environment is properly installed
3. Test with a small number of rollouts first

### Custom Evaluation Configs

Create new evaluation configuration files:

```yaml
# custom_eval.yaml
evaluation:
  environments:
    - name: "MyCustomEnv"
      n_rollouts: 50
      episode_length: 10
      env_name: myenv
      # ... other config
```

Then use it with:

```bash
./eval_wrapper.sh checkpoint /path/to/model custom_eval "my_custom_eval"
```

## Performance Tips

1. **Use Multiple GPUs**: For faster evaluation, use multiple GPUs
2. **Adjust Rollouts**: Reduce `n_rollouts` for faster evaluation
3. **Batch Processing**: Run multiple evaluations in parallel on different nodes
4. **Memory Management**: Monitor memory usage and adjust batch sizes accordingly
