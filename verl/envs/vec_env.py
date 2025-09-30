# Adapted from https://github.com/zoeyuchao/mappo/blob/main/onpolicy/envs/env_wrappers.py under the MIT License.
# Original author: yuchao

import numpy as np
import torch
from multiprocessing import Process, Pipe
import random

from collections import defaultdict

def merge_metrics(infos):
    merged_infos = defaultdict(list)
    for info in infos:
        for key, value in info["metrics"].items():
            merged_infos[key].append(value)
    merged_infos = {key: np.mean(values) for key, values in merged_infos.items()}
    return merged_infos


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv:
    
    def __init__(self, env_name, config, env_fns, captioner_fns):
        
        self.config = config
        self.n_rollouts = config.envs.n_rollouts
        assert len(env_fns) == self.n_rollouts, "Number of env_fns must match n_rollouts"
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_rollouts)])
        self.processes = []
        for rank, (work_remote, remote, env_fn, captioner_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns, captioner_fns)):
            p = Process(
                target=worker,
                args=(rank, work_remote, remote, env_name, CloudpickleWrapper(env_fn), CloudpickleWrapper(captioner_fn)),
            )
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            self.processes.append(p)
        
        for remote in self.work_remotes:
            remote.close()
        
        
        # Cache for storing last known state of environments (for skip functionality)
        self.last_obs = [None] * self.n_rollouts
        self.last_reward = [0.0] * self.n_rollouts
        self.last_terminated = [False] * self.n_rollouts
        self.last_truncated = [False] * self.n_rollouts
        self.last_info = [{"metrics": {}}] * self.n_rollouts
            
    def step(self, actions):
        # Handle skip actions for frozen environments
        skip_action = "__SKIP__"
        active_remotes = []
        active_actions = []
        skip_indices = []
        
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            if action == skip_action:
                skip_indices.append(i)
            else:
                active_remotes.append(remote)
                active_actions.append(action)
        
        # Only send step commands to active environments
        for i, (remote, action) in enumerate(zip(active_remotes, active_actions)):
            try:
                remote.send(('step', action))
            except BrokenPipeError as e:
                print(f"[ERROR] VecEnv.step: BrokenPipeError when sending to worker {i}: {e}")
                raise RuntimeError("Worker process connection broken during step. Check for errors in worker processes.")
        
        # Collect results from active environments
        active_results = []
        for i, remote in enumerate(active_remotes):
            try:
                result = remote.recv()
                # Check if worker sent an error
                if isinstance(result, tuple) and len(result) > 0 and result[0] == 'error':
                    print(f"[ERROR] VecEnv.step: Worker {i} sent error: {result[1]}")
                    raise RuntimeError(f"Worker {i} error: {result[1]}")
                
                active_results.append(result)
            except Exception as e:
                print(f"[ERROR] VecEnv.step: Exception receiving from worker {i}: {e}")
                raise
        
        # Reconstruct full results with cached data for skipped environments
        full_results = []
        active_idx = 0
        for i in range(len(self.remotes)):
            if i in skip_indices:
                # Return cached data for skipped environments
                full_results.append((
                    self.last_obs[i],
                    self.last_reward[i],
                    self.last_terminated[i],
                    self.last_truncated[i],
                    self.last_info[i]
                ))
            else:
                full_results.append(active_results[active_idx])
                active_idx += 1
        
        obs, rews, terminated, truncated, infos = zip(*full_results)

        
        # Update cache with current results
        for i in range(len(self.remotes)):
            self.last_obs[i] = obs[i]
            self.last_reward[i] = rews[i]
            self.last_terminated[i] = terminated[i]
            self.last_truncated[i] = truncated[i]
            self.last_info[i] = infos[i]
        
        infos = merge_metrics(infos)
        
        return obs, np.stack(rews), np.stack(terminated), np.stack(truncated), infos
    
    def reset(self, seed=None, seed_group_size=None, use_incremental_seeds=False):
        """
        Reset all environments with seeds.
        
        Args:
            seed: Base seed for resetting environments
            seed_group_size: If provided, group successive sets of this many rollouts to have the same seed.
                           Each group gets seed + group_index. Must divide n_rollouts evenly.
            use_incremental_seeds: If True, each worker gets seed + worker_rank (for evaluation).
                                 If False, all workers get the same seed (for GRPO training).
        """
        if seed_group_size is not None and self.n_rollouts % seed_group_size != 0:
            raise ValueError("n_rollouts must be divisible by seed_group_size")

        for i, remote in enumerate(self.remotes):
            try:
                if use_incremental_seeds and seed is not None:
                    # Create unique seed for each worker: base_seed + worker_rank (for evaluation)
                    worker_seed = seed + i
                elif seed_group_size is not None and seed is not None:
                    # Group successive sets of seed_group_size rollouts to have the same seed
                    group_index = i // seed_group_size
                    worker_seed = seed + group_index
                else:
                    # Use the same seed for all workers (for GRPO training)
                    worker_seed = seed
                remote.send(('reset', worker_seed))
            except Exception as e:
                raise RuntimeError(f"Error resetting environment {i}: {e}")
        
        observations, infos = zip(*[remote.recv() for remote in self.remotes])
        
        # Update cache with reset results
        for i in range(len(self.remotes)):
            self.last_obs[i] = observations[i]
            self.last_reward[i] = 0.0
            self.last_terminated[i] = False
            self.last_truncated[i] = False
            self.last_info[i] = infos[i]
        
        return observations, infos
    
    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        images = [remote.recv() for remote in self.remotes]
        return images

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
 

    
def worker(rank, remote, parent_remote, env_name, env_fn_wrapper, captioner_fn_wrapper):
    
    random.seed(rank)
    np.random.seed(rank)
    
    parent_remote.close()
    try:
        env = env_fn_wrapper.x()
    except Exception as e:
        print(f"[ERROR] Worker {rank}: Failed to create environment: {e}")
        raise
    
    try:
        captioner = captioner_fn_wrapper.x()
    except Exception as e:
        print(f"[ERROR] Worker {rank}: Failed to create captioner: {e}")
        raise
    
    image = None
    
    def env_step(action):
        try:
            full_action, executed_action, is_valid, metrics = env.extract_action(action)
            
            env_obs, reward, terminated, truncated, info = env.step(executed_action, is_valid)
            
            image = env_obs.get("image", None)
            instructions = env_obs["mission"]  if env_name == "babyai" else None
            inst_prompt = env.get_instruction_prompt(instructions=instructions, info=info)
            captioner.prompt_builder.update_instruction_prompt(inst_prompt)
            captioner.update_action(full_action, executed_action)
            info["metrics"] = metrics
            return captioner.get_obs(env_obs), reward, terminated, truncated, info, image
        except Exception as e:
            print(f"[ERROR] Worker {rank}: Exception in env_step: {e}")
            import traceback
            print(f"[ERROR] Worker {rank}: Traceback: {traceback.format_exc()}")
            raise

    def env_reset(seed=None):
        captioner.reset()
        env_obs, info = env.reset(seed=seed)
        image = env_obs.get("image", None)
        instructions = env_obs["mission"]  if env_name == "babyai" else None
        inst_prompt = env.get_instruction_prompt(instructions=instructions)
        captioner.prompt_builder.update_instruction_prompt(inst_prompt)
        return captioner.get_obs(env_obs), info, image
        
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, terminated, truncated, info, image = env_step(data)
                if terminated or truncated:
                    obs, _, image = env_reset(seed=None)  
                remote.send((obs, reward, terminated, truncated, info))
                
            elif cmd == 'reset':
                seed = data
                obs, info, image = env_reset(seed=seed)
                remote.send((obs, info))
                
            elif cmd == 'render':
                remote.send(image)
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                print(f"[ERROR] Worker {rank}: Unknown command: '{cmd}'")
                raise NotImplementedError
                
        except Exception as e:
            print(f"[ERROR] Worker {rank}: Exception in command loop: {e}")
            import traceback
            print(f"[ERROR] Worker {rank}: Traceback: {traceback.format_exc()}")
            # Send error back to main process
            try:
                remote.send(('error', str(e)))
            except Exception as send_error:
                print(f"[ERROR] Worker {rank}: Failed to send error to main process: {send_error}")
                pass  # If we can't send the error, just exit
            break

        
  
    
