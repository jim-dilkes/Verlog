#!/usr/bin/env python3
"""
Standalone Evaluation Script for VERL Models

This script allows you to run evaluation suites on pre-existing checkpoints or base LLMs
without needing to run the full training pipeline. It uses the MultiEnvEvaluator to run
evaluations across multiple environments as specified in the evaluation configuration.

Usage:
    python run_evaluation.py --model_path <path> --eval_config <config> [options]

Example:
    python run_evaluation.py --model_path /path/to/checkpoint --eval_config eval_1 --wandb_project AAMAS_msrl_base
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from pprint import pprint

import ray
import torch
import numpy as np
import psutil
from omegaconf import OmegaConf, open_dict
from codetiming import Timer

# Add the verl package to the path and suppress warnings
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from verl.utils.warning_suppression import setup_warning_suppression
setup_warning_suppression()

from verl.trainer.ppo.multi_env_evaluator import MultiEnvEvaluator
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking


class StandaloneEvaluator:
    """
    Standalone evaluator that can run evaluation suites on pre-existing models or checkpoints.
    """
    
    def __init__(self, config_path: str, model_path: str, eval_config_name: str, 
                 wandb_project: str = "AAMAS_msrl_base", wandb_run_name: Optional[str] = None,
                 output_dir: str = "./evaluation_results", ppo_config_override: str = "Verlog/evaluation/eval_config.yaml",
                 save_config_path: Optional[str] = None):
        """
        Initialize the standalone evaluator.
        
        Args:
            config_path: Path to the base configuration file
            model_path: Path to the model checkpoint or base model
            eval_config_name: Name of the evaluation configuration (e.g., 'eval_1')
            wandb_project: Wandb project name for logging
            wandb_run_name: Custom run name for wandb (optional)
            output_dir: Directory to save evaluation results
            ppo_config_override: Path to evaluation configuration file
            save_config_path: Path to save final configuration (optional)
        """
        self.model_path = model_path
        self.eval_config_name = eval_config_name
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ppo_config_override = ppo_config_override
        self.config_path = config_path
        self.save_config_path = save_config_path
        print(f"DEBUG: ppo_config_override set to: {self.ppo_config_override}")
        
        # Load base configuration
        self.config = self._load_config(config_path)
        
        # Load evaluation configuration
        self._load_eval_config()
        
        # Load evaluation parameters configuration
        self.eval_params = self._load_eval_params_config()
        
        # Initialize tokenizer and processor
        self.tokenizer = None
        self.processor = None
        
        # Initialize Ray workers
        self.actor_rollout_wg = None
        self.multi_env_evaluator = None
        
        # Results storage
        self.results = {}
        
    def _load_config(self, config_path: str) -> OmegaConf:
        """Load the base configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = OmegaConf.load(config_path)
        print(f"Loaded base configuration from: {config_path}")
        return config
    
    def _load_eval_config(self):
        """Load the evaluation configuration."""
        # Always use Verlog/ prefix since we're running from the verlog directory
        eval_config_path = f"Verlog/verl/trainer/config/evaluation/{self.eval_config_name}.yaml"
        
        if not os.path.exists(eval_config_path):
            raise FileNotFoundError(f"Evaluation configuration not found: {eval_config_path}")
        
        eval_config = OmegaConf.load(eval_config_path)
        
        # Merge evaluation config into main config
        with open_dict(self.config):
            self.config.evaluation = eval_config.evaluation
        
        print(f"Loaded evaluation configuration: {self.eval_config_name}")
        print(f"Number of evaluation environments: {len(self.config.evaluation.environments)}")
        
        # Print environment names
        for i, env_config in enumerate(self.config.evaluation.environments):
            env_name = env_config.get('name', f'env_{i}')
            print(f"  Environment {i}: {env_name}")
    
    def _load_eval_params_config(self) -> OmegaConf:
        """Load the evaluation parameters configuration."""
        print(f"DEBUG: _load_eval_params_config called with self.ppo_config_override: {self.ppo_config_override}")
        if not os.path.exists(self.ppo_config_override):
            print(f"Warning: Evaluation config file not found: {self.ppo_config_override}")
            print("Using default evaluation parameters...")
            return OmegaConf.create({})
        
        eval_params = OmegaConf.load(self.ppo_config_override)
        print(f"Loaded evaluation parameters from: {self.ppo_config_override}")
        print(f"DEBUG: eval_params.get('gpu', {{}}): {eval_params.get('gpu', {})}")
        return eval_params
    
    def _setup_model_config(self):
        """Set up model configuration for evaluation."""
        # Download model if it's a remote path
        local_model_path = copy_to_local(self.model_path)
        
        # Update model path in config
        with open_dict(self.config):
            self.config.actor_rollout_ref.model.path = local_model_path
            self.config.critic.model.path = local_model_path
            self.config.critic.model.tokenizer_path = local_model_path
        
        # Set evaluation-specific configurations
        with open_dict(self.config):
            # Disable training-specific features
            eval_settings = self.eval_params.get('evaluation', {})
            self.config.trainer.val_before_train = eval_settings.get('val_before_train', False)
            self.config.trainer.test_freq = eval_settings.get('test_freq', -1)
            self.config.trainer.save_freq = eval_settings.get('save_freq', -1)
            self.config.trainer.total_epochs = eval_settings.get('total_epochs', 1)
            self.config.trainer.log_val_generations = eval_settings.get('log_val_generations', 1)
            self.config.trainer.logger = eval_settings.get('logger', ['console', 'wandb'])
            
            # Set wandb project
            self.config.trainer.project_name = self.wandb_project
            if self.wandb_run_name:
                self.config.trainer.experiment_name = self.wandb_run_name
            else:
                # Generate run name from model path and timestamp
                model_name = Path(self.model_path).name
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.config.trainer.experiment_name = f"eval_{model_name}_{timestamp}"
            
            # Set output directory
            self.config.trainer.default_local_dir = str(self.output_dir)
            
            # Configure generation parameters from eval_config.yaml
            gen_params = self.eval_params.get('model', {}).get('generation', {})
            self.config.actor_rollout_ref.rollout.val_kwargs.do_sample = gen_params.get('do_sample', False)
            self.config.actor_rollout_ref.rollout.val_kwargs.temperature = gen_params.get('temperature', 0.0)
            self.config.actor_rollout_ref.rollout.val_kwargs.top_k = gen_params.get('top_k', -1)
            self.config.actor_rollout_ref.rollout.val_kwargs.top_p = gen_params.get('top_p', 1.0)
            
            # Set response and prompt lengths
            self.config.data.max_response_length = gen_params.get('max_response_length', 256)
            self.config.data.max_prompt_length = gen_params.get('max_prompt_length', 1500)
            
            # Configure model loading parameters from eval_config.yaml
            loading_params = self.eval_params.get('model', {}).get('loading', {})
            # Always use bfloat16 for model dtype
            torch_dtype = 'bfloat16'
            
            # Keep torch_dtype as string for OmegaConf compatibility
            # The model loading code will handle the conversion to actual torch dtypes
            self.config.actor_rollout_ref.model.torch_dtype = torch_dtype
            self.config.critic.model.torch_dtype = torch_dtype
            
            # Also set dtype for VLLM rollout to ensure bfloat16 is used
            self.config.actor_rollout_ref.rollout.dtype = torch_dtype
            
            # Configure GPU settings from eval_config.yaml
            gpu_params = self.eval_params.get('gpu', {})
            print(f"DEBUG: GPU params from eval_config: {gpu_params}")
            if gpu_params:
                # For large models like 32B, reduce memory utilization to leave room for KV cache
                memory_utilization = gpu_params.get('memory_utilization', 0.7)  # Reduced from 0.95 to 0.7
                print(f"DEBUG: Setting gpu_memory_utilization to: {memory_utilization}")
                self.config.actor_rollout_ref.rollout.gpu_memory_utilization = memory_utilization
                print(f"Set GPU memory utilization to: {memory_utilization} (reduced for large model compatibility)")
                
                # Set additional vLLM memory optimization parameters
                if 'max_model_len' in gpu_params:
                    self.config.actor_rollout_ref.rollout.max_model_len = gpu_params['max_model_len']
                    print(f"Set max_model_len to: {gpu_params['max_model_len']}")
                
                if 'swap_space' in gpu_params:
                    self.config.actor_rollout_ref.rollout.swap_space = gpu_params['swap_space']
                    print(f"Set swap_space to: {gpu_params['swap_space']}")
                
                if 'block_size' in gpu_params:
                    self.config.actor_rollout_ref.rollout.block_size = gpu_params['block_size']
                    print(f"Set block_size to: {gpu_params['block_size']}")
                    
            else:
                print(f"DEBUG: No GPU params found, using default from base config: {self.config.actor_rollout_ref.rollout.gpu_memory_utilization}")
            
            # Configure environment settings from eval_config.yaml
            env_params = self.eval_params.get('environment', {})
            if env_params:
                self.config.envs.n_rollouts = env_params.get('n_rollouts', 16)
                print(f"Set number of rollouts to: {env_params.get('n_rollouts', 16)}")
        
        print(f"Model path set to: {local_model_path}")
        print(f"Wandb project: {self.wandb_project}")
        print(f"Run name: {self.config.trainer.experiment_name}")
        
        # Print final configuration for verification
        self._print_final_config()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer and processor."""
        print("Initializing tokenizer and processor...")
        
        # Get loading parameters from eval_config.yaml
        loading_params = self.eval_params.get('model', {}).get('loading', {})
        trust_remote_code = loading_params.get('trust_remote_code', False)
        use_fast_tokenizer = loading_params.get('use_fast_tokenizer', True)
        
        self.tokenizer = hf_tokenizer(
            self.config.actor_rollout_ref.model.path, 
            trust_remote_code=trust_remote_code
        )
        self.processor = hf_processor(
            self.config.actor_rollout_ref.model.path, 
            use_fast=use_fast_tokenizer
        )
        
        print("Tokenizer and processor initialized successfully")
    
    def _print_final_config(self):
        """Print the final configuration for verification."""
        print("\n" + "="*80)
        print("FINAL EVALUATION CONFIGURATION")
        print("="*80)
        
        # Model configuration
        print("\n📁 MODEL CONFIGURATION:")
        print(f"  Model path: {self.config.actor_rollout_ref.model.path}")
        print(f"  Trust remote code: {self.eval_params.get('model', {}).get('loading', {}).get('trust_remote_code', False)}")
        print(f"  Use fast tokenizer: {self.eval_params.get('model', {}).get('loading', {}).get('use_fast_tokenizer', True)}")
        
        # Generation parameters
        gen_params = self.eval_params.get('model', {}).get('generation', {})
        print("\n🎯 GENERATION PARAMETERS:")
        print(f"  Temperature: {gen_params.get('temperature', 0.0)}")
        print(f"  Top-k: {gen_params.get('top_k', -1)}")
        print(f"  Top-p: {gen_params.get('top_p', 1.0)}")
        print(f"  Do sample: {gen_params.get('do_sample', False)}")
        print(f"  Max response length: {gen_params.get('max_response_length', 256)}")
        print(f"  Max prompt length: {gen_params.get('max_prompt_length', 1500)}")
        
        # Evaluation settings
        eval_settings = self.eval_params.get('evaluation', {})
        print("\n⚙️  EVALUATION SETTINGS:")
        print(f"  Val before train: {eval_settings.get('val_before_train', False)}")
        print(f"  Test frequency: {eval_settings.get('test_freq', -1)}")
        print(f"  Save frequency: {eval_settings.get('save_freq', -1)}")
        print(f"  Total epochs: {eval_settings.get('total_epochs', 1)}")
        print(f"  Log val generations: {eval_settings.get('log_val_generations', 1)}")
        print(f"  Logger backends: {eval_settings.get('logger', ['console', 'wandb'])}")
        
        # Training configuration
        print("\n🏋️  TRAINING CONFIGURATION:")
        print(f"  GPUs per node: {self.config.trainer.n_gpus_per_node}")
        print(f"  Number of nodes: {self.config.trainer.nnodes}")
        print(f"  Tensor model parallel size: {self.config.actor_rollout_ref.rollout.tensor_model_parallel_size}")
        print(f"  Project name: {self.config.trainer.project_name}")
        print(f"  Experiment name: {self.config.trainer.experiment_name}")
        print(f"  Output directory: {self.config.trainer.default_local_dir}")
        
        # Environment configuration
        print("\n🌍 ENVIRONMENT CONFIGURATION:")
        print(f"  Number of environments: {len(self.config.evaluation.environments)}")
        for i, env_config in enumerate(self.config.evaluation.environments):
            env_name = env_config.get('name', f'env_{i}')
            print(f"    {i+1}. {env_name}")
            print(f"       - Rollouts: {env_config.get('n_rollouts', 'N/A')}")
            print(f"       - Episode length: {env_config.get('episode_length', 'N/A')}")
            print(f"       - Environment: {env_config.get('env_name', 'N/A')}")
        
        # Config file information
        print("\n📄 CONFIGURATION FILES:")
        print(f"  Base config: {self.config_path}")
        print(f"  Eval config: {self.eval_config_name}")
        print(f"  Eval params file: {self.ppo_config_override}")
        
        print("\n" + "="*80)
        print("CONFIGURATION VERIFICATION COMPLETE")
        print("="*80 + "\n")
        
        # Save configuration to file if requested
        if self.save_config_path:
            self._save_config_to_file()
    
    def _save_config_to_file(self):
        """Save the final configuration to a file for record-keeping."""
        import json
        from datetime import datetime
        
        config_dict = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "eval_config_name": self.eval_config_name,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "output_dir": str(self.output_dir),
            "config_files": {
                "base_config": self.config_path,
                "eval_config": self.eval_config_name,
                "eval_params_file": self.ppo_config_override
            },
            "model_config": {
                "path": self.config.actor_rollout_ref.model.path,
                "trust_remote_code": self.eval_params.get('model', {}).get('loading', {}).get('trust_remote_code', False),
                "use_fast_tokenizer": self.eval_params.get('model', {}).get('loading', {}).get('use_fast_tokenizer', True)
            },
            "generation_params": self.eval_params.get('model', {}).get('generation', {}),
            "evaluation_settings": self.eval_params.get('evaluation', {}),
            "training_config": {
                "gpus_per_node": self.config.trainer.n_gpus_per_node,
                "nnodes": self.config.trainer.nnodes,
                "tensor_model_parallel_size": self.config.actor_rollout_ref.rollout.tensor_model_parallel_size,
                "project_name": self.config.trainer.project_name,
                "experiment_name": self.config.trainer.experiment_name,
                "output_directory": self.config.trainer.default_local_dir
            },
            "environments": [
                {
                    "name": env_config.get('name', f'env_{i}'),
                    "rollouts": env_config.get('n_rollouts', 'N/A'),
                    "episode_length": env_config.get('episode_length', 'N/A'),
                    "env_name": env_config.get('env_name', 'N/A')
                }
                for i, env_config in enumerate(self.config.evaluation.environments)
            ]
        }
        
        with open(self.save_config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"📄 Configuration saved to: {self.save_config_path}")
    
    def _monitor_ray_processes(self):
        """Monitor Ray processes and provide diagnostics."""
        print("Monitoring Ray processes...")
        try:
            import subprocess
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            ray_processes = [line for line in result.stdout.split('\n') if 'ray' in line.lower()]
            print(f"Found {len(ray_processes)} Ray-related processes:")
            for proc in ray_processes:
                print(f"  {proc}")
        except Exception as e:
            print(f"Failed to monitor Ray processes: {e}")
        
        # Check Ray cluster status
        try:
            if ray.is_initialized():
                print(f"Ray cluster resources: {ray.cluster_resources()}")
                print(f"Ray available resources: {ray.available_resources()}")
                print(f"Ray nodes: {ray.nodes()}")
            else:
                print("Ray is not initialized")
        except Exception as e:
            print(f"Failed to get Ray cluster status: {e}")

    def _initialize_ray_workers(self):
        """Initialize Ray workers for model inference."""
        print("=" * 80)
        print("RAY WORKER INITIALIZATION DEBUG")
        print("=" * 80)
        
        print("Starting Ray worker initialization...")
        print(f"Ray is initialized: {ray.is_initialized()}")
        print(f"Ray cluster resources: {ray.cluster_resources()}")
        print(f"Ray available resources: {ray.available_resources()}")
        
        # Monitor Ray processes before starting
        self._monitor_ray_processes()
        
        # Check system resources before worker creation
        print(f"System memory available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
        print(f"System memory used: {psutil.virtual_memory().percent}%")
        
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Define worker classes based on strategy
        print(f"Using strategy: {self.config.actor_rollout_ref.actor.strategy}")
        try:
            if self.config.actor_rollout_ref.actor.strategy == 'fsdp':
                print("Importing FSDP workers...")
                from verl.workers.fsdp_workers import ActorRolloutRefWorker
                ray_worker_group_cls = RayWorkerGroup
                print("FSDP workers imported successfully")
            elif self.config.actor_rollout_ref.actor.strategy == 'megatron':
                print("Importing Megatron workers...")
                from verl.workers.megatron_workers import ActorRolloutRefWorker
                from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
                ray_worker_group_cls = NVMegatronRayWorkerGroup
                print("Megatron workers imported successfully")
            else:
                raise NotImplementedError(f"Strategy {self.config.actor_rollout_ref.actor.strategy} not supported")
        except Exception as e:
            print(f"Failed to import workers: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Set up resource pool
        print("Setting up resource pool...")
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }
        
        print(f"Resource pool spec: {resource_pool_spec}")
        print(f"Resource mapping: {mapping}")
        print(f"GPUs per node: {self.config.trainer.n_gpus_per_node}")
        print(f"Number of nodes: {self.config.trainer.nnodes}")
        
        try:
            resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resource_pool_spec, 
                mapping=mapping
            )
            print("Resource pool manager created successfully")
        except Exception as e:
            print(f"Failed to create resource pool manager: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create worker group
        print("Creating worker group...")
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }
        
        try:
            print("Creating resource pool...")
            resource_pool_manager.create_resource_pool()
            print("Resource pool created successfully")
        except Exception as e:
            print(f"Failed to create resource pool: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create actor rollout worker
        print("Getting resource pool for ActorRollout...")
        try:
            resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
            print(f"Resource pool obtained: {resource_pool}")
        except Exception as e:
            print(f"Failed to get resource pool: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("Creating Ray remote worker class...")
        try:
            actor_rollout_cls = ray.remote(ActorRolloutRefWorker)
            print("Ray remote worker class created successfully")
        except Exception as e:
            print(f"Failed to create Ray remote worker class: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("Importing Ray controller classes...")
        try:
            from verl.single_controller.ray import RayClassWithInitArgs
            from verl.single_controller.ray.base import create_colocated_worker_cls
            print("Ray controller classes imported successfully")
        except Exception as e:
            print(f"Failed to import Ray controller classes: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("Creating RayClassWithInitArgs...")
        try:
            actor_rollout_cls_with_args = RayClassWithInitArgs(
                cls=actor_rollout_cls,
                config=self.config.actor_rollout_ref,
                role='actor_rollout'
            )
            print("RayClassWithInitArgs created successfully")
        except Exception as e:
            print(f"Failed to create RayClassWithInitArgs: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("Creating colocated worker class...")
        try:
            worker_dict_cls = create_colocated_worker_cls(
                class_dict={'actor_rollout': actor_rollout_cls_with_args}
            )
            print("Colocated worker class created successfully")
        except Exception as e:
            print(f"Failed to create colocated worker class: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Set up kwargs for RayWorkerGroup (same as trainer)
        print("Setting up worker group kwargs...")
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        print(f"Worker group kwargs: {wg_kwargs}")
        
        print("Creating Ray worker group...")
        try:
            wg_dict = ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs
            )
            print("Ray worker group created successfully")
        except Exception as e:
            print(f"Failed to create Ray worker group: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("Spawning worker group...")
        try:
            # Add timeout to catch hanging operations
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Worker group spawn timed out after 60 seconds")
            
            # Set timeout for worker spawning
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
            
            print("Calling wg_dict.spawn()...")
            spawn_wg = wg_dict.spawn(prefix_set={'actor_rollout'})
            signal.alarm(0)  # Cancel timeout
            
            print("Worker group spawned successfully")
            print("Getting actor_rollout from spawn_wg...")
            self.actor_rollout_wg = spawn_wg['actor_rollout']
            print("Actor rollout worker group assigned successfully")
            
            # Monitor Ray processes after successful spawning
            print("Monitoring Ray processes after worker spawning...")
            self._monitor_ray_processes()
            
        except TimeoutError as e:
            signal.alarm(0)  # Cancel timeout
            print(f"Worker group spawn timed out: {e}")
            print("This suggests the Ray worker registration is hanging")
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            print(f"Failed to spawn worker group: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        
        # Keep reference to wg_dict to prevent Ray from garbage collecting it
        # This is critical for preventing "ActorDiedError" - same as trainer does
        self.wg_dicts = [wg_dict]
        
        # Initialize model
        print("=" * 80)
        print("MODEL INITIALIZATION DEBUG")
        print("=" * 80)
        
        print("Starting model initialization...")
        
        # Print memory usage before model initialization
        print(f"System memory before model init: {psutil.virtual_memory().available / 1024**3:.2f} GB available")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} before model init - Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        
        print("Calling actor_rollout_wg.init_model()...")
        try:
            self.actor_rollout_wg.init_model()
            print("Model initialization completed successfully")
        except Exception as e:
            print(f"Model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Print memory usage after model initialization
        print(f"System memory after model init: {psutil.virtual_memory().available / 1024**3:.2f} GB available")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} after model init - Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        
        # Load checkpoint if it's a checkpoint path
        if self._is_checkpoint_path(self.model_path):
            print(f"Loading checkpoint from: {self.model_path}")
            try:
                self.actor_rollout_wg.load_checkpoint(self.model_path)
                print("Checkpoint loaded successfully")
            except Exception as e:
                print(f"Checkpoint loading failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        print("Ray workers initialized successfully")
    
    def _is_checkpoint_path(self, path: str) -> bool:
        """Check if the path points to a checkpoint directory."""
        path = Path(path)
        return path.is_dir() and any((path / "actor").exists(), (path / "model.safetensors").exists())
    
    def _initialize_evaluator(self):
        """Initialize the multi-environment evaluator."""
        print("Initializing multi-environment evaluator...")
        
        self.multi_env_evaluator = MultiEnvEvaluator(
            config=self.config,
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            val_reward_fn=None,  # We don't need reward function for evaluation
            eval_config=self.config.evaluation
        )
        
        print("Multi-environment evaluator initialized successfully")
    
    def _setup_wandb_logging(self):
        """Set up wandb logging."""
        print("Setting up wandb logging...")
        
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=['console', 'wandb'],
            config=OmegaConf.to_container(self.config, resolve=True),
            group=f"evaluation_{self.eval_config_name}"
        )
        
        print(f"Wandb logging configured for project: {self.wandb_project}")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation suite.
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Run evaluation
            print("Running evaluation across all environments...")
            evaluation_metrics = self.multi_env_evaluator.evaluate(global_step=0)
            
            # Log results to wandb
            print("Logging results to wandb...")
            self.logger.log(data=evaluation_metrics, step=0)
            
            # Store results
            self.results = {
                'evaluation_metrics': evaluation_metrics,
                'model_path': self.model_path,
                'eval_config': self.eval_config_name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration_seconds': time.time() - start_time
            }
            
            # Save results to file
            self._save_results()
            
            print("=" * 80)
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Total evaluation time: {self.results['duration_seconds']:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            print(f"Wandb run: {self.wandb_project}/{self.config.trainer.experiment_name}")
            
            # Print summary of results
            self._print_results_summary()
            
            return self.results
            
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self):
        """Save evaluation results to file."""
        results_file = self.output_dir / f"evaluation_results_{self.config.trainer.experiment_name}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def _print_results_summary(self):
        """Print a summary of the evaluation results."""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        metrics = self.results['evaluation_metrics']
        
        # Group metrics by environment
        env_metrics = {}
        for key, value in metrics.items():
            if key.startswith('eval_'):
                parts = key.split('/')
                if len(parts) >= 2:
                    env_name = parts[0].replace('eval_', '')
                    metric_name = '/'.join(parts[1:])
                    
                    if env_name not in env_metrics:
                        env_metrics[env_name] = {}
                    env_metrics[env_name][metric_name] = value
        
        for env_name, env_data in env_metrics.items():
            print(f"\n{env_name}:")
            for metric_name, value in env_data.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'wg_dicts') and self.wg_dicts is not None:
            try:
                # Clean up Ray worker groups (same as trainer)
                for wg_dict in self.wg_dicts:
                    ray.kill(wg_dict)
            except:
                pass
        
        if self.actor_rollout_wg is not None:
            try:
                # Clean up Ray workers
                ray.kill(self.actor_rollout_wg)
            except:
                pass


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description="Run evaluation suite on VERL models")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint or base model")
    parser.add_argument("--eval_config", type=str, default="eval_1",
                       help="Evaluation configuration name (default: eval_1)")
    parser.add_argument("--config_path", type=str, 
                       default="Verlog/verl/trainer/config/ppo_trainer.yaml",
                       help="Path to base configuration file")
    parser.add_argument("--wandb_project", type=str, default="AAMAS_msrl_base",
                       help="Wandb project name (default: AAMAS_msrl_base)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Custom wandb run name (optional)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results (default: ./evaluation_results)")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use (default: 1)")
    parser.add_argument("--nodes", type=int, default=1,
                       help="Number of nodes to use (default: 1)")
    parser.add_argument("--ppo_config_override", type=str, default="Verlog/evaluation/eval_config.yaml",
                       help="Path to evaluation configuration file (default: Verlog/evaluation/eval_config.yaml)")
    parser.add_argument("--save_config", type=str, default=None,
                       help="Save final configuration to file (optional)")
    
    args = parser.parse_args()
    print(f"DEBUG: Parsed args.ppo_config_override: {args.ppo_config_override}")
    
    # Initialize Ray with comprehensive debugging
    print("=" * 80)
    print("RAY INITIALIZATION DEBUG")
    print("=" * 80)
    
    if ray.is_initialized():
        print("Ray is already initialized, shutting down first...")
        try:
            ray.shutdown()
            print("Ray shutdown successful")
        except Exception as e:
            print(f"Ray shutdown failed: {e}")
    
    print("Starting Ray initialization...")
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"Available CPUs: {os.cpu_count()}")
    
    try:
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN'
                }
            },
            # Reduce object store memory to prevent OOM
            object_store_memory=2000000000,  # 2GB instead of 50GB
            # Set num_cpus to prevent resource conflicts
            num_cpus=min(4, os.cpu_count()),
            # Disable dashboard to reduce overhead
            include_dashboard=False,
            # Set log level to reduce noise
            log_to_driver=False
        )
        print("Ray initialization successful!")
        print(f"Ray cluster resources: {ray.cluster_resources()}")
        print(f"Ray available resources: {ray.available_resources()}")
    except Exception as e:
        print(f"Ray initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    try:
        # Create evaluator
        evaluator = StandaloneEvaluator(
            config_path=args.config_path,
            model_path=args.model_path,
            eval_config_name=args.eval_config,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            output_dir=args.output_dir,
            ppo_config_override=args.ppo_config_override,
            save_config_path=args.save_config
        )
        
        # Update GPU configuration
        with open_dict(evaluator.config):
            evaluator.config.trainer.n_gpus_per_node = args.gpus
            evaluator.config.trainer.nnodes = args.nodes
            
            # Fix tensor model parallel size to match available GPUs
            evaluator.config.actor_rollout_ref.rollout.tensor_model_parallel_size = args.gpus
            print(f"Set tensor_model_parallel_size to {args.gpus} to match available GPUs")
        
        # Run evaluation pipeline
        evaluator._setup_model_config()
        evaluator._initialize_tokenizer()
        evaluator._initialize_ray_workers()
        evaluator._initialize_evaluator()
        evaluator._setup_wandb_logging()
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Cleanup
        evaluator.cleanup()
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results: {results['evaluation_metrics']}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
