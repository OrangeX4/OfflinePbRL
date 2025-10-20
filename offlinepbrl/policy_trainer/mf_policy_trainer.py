import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinepbrl.buffer import ReplayBuffer, PrefBuffer, TrajectoryBuffer
from offlinepbrl.utils.logger import Logger
from offlinepbrl.policy import BasePolicy


# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        pref_buffer: Optional[PrefBuffer] = None,
        pref_batch_size: Optional[int] = None,
        pref_batch_num: Optional[int] = None,
        eval_freq: int = 1,
        traj_batch_size: Optional[int] = None,
        segment_size: Optional[int] = None,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        self.pref_buffer = pref_buffer
        self.pref_batch_size = pref_batch_size if pref_batch_size is not None else batch_size
        self.pref_batch_num = pref_batch_num

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self._eval_freq = eval_freq
        self._traj_batch_size = traj_batch_size
        self._segment_size = segment_size
        self._supports_traj_sampling = isinstance(buffer, TrajectoryBuffer)

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                # Sample from both buffers if preference buffer is available
                if self.pref_buffer is not None:
                    replay_batch = self.buffer.sample(self._batch_size)
                    if self.pref_batch_num is not None:
                        preference_batches = [self.pref_buffer.sample(self.pref_batch_size) for _ in range(self.pref_batch_num)]
                        batch = {
                            "replay": replay_batch,
                            "pref": preference_batches,
                        }
                    else:
                        preference_batch = self.pref_buffer.sample(self.pref_batch_size)
                        batch = {
                            "replay": replay_batch,
                            "pref": preference_batch,
                        }
                    
                    # Add trajectory batch if buffer supports it and traj_batch_size is set
                    if self._traj_batch_size is not None:
                        if self._supports_traj_sampling:
                            # TrajectoryBuffer: sample complete trajectories
                            traj_batch = self.buffer.sample_trajectories(self._traj_batch_size)
                            batch["trajectory"] = traj_batch
                        elif self._segment_size is not None:
                            # ReplayBuffer with segment sampling (for APPO)
                            traj_batch = self.buffer.sample_trajectory(self._traj_batch_size, self._segment_size)
                            batch["traj"] = traj_batch
                else:
                    batch = self.buffer.sample(self._batch_size)
                    
                    # Add trajectory batch if buffer supports it and traj_batch_size is set
                    if self._traj_batch_size is not None:
                        if self._supports_traj_sampling:
                            # TrajectoryBuffer: sample complete trajectories
                            traj_batch = self.buffer.sample_trajectories(self._traj_batch_size)
                            batch = {
                                "replay": batch,
                                "trajectory": traj_batch,
                            }
                        elif self._segment_size is not None:
                            # ReplayBuffer with segment sampling (for APPO)
                            traj_batch = self.buffer.sample_trajectory(self._traj_batch_size, self._segment_size)
                            batch = {
                                "replay": batch,
                                "traj": traj_batch,
                            }
                
                loss = self.policy.learn(batch, epoch=e, step=it)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            if e % self._eval_freq == 0:
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                
                # Check if environment has normalized score method (D4RL environments)
                if hasattr(self.eval_env, 'get_normalized_score'):
                    norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                    norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                    last_10_performance.append(norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                    self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                else:
                    # For environments without normalized score (e.g., MetaWorld)
                    last_10_performance.append(ep_reward_mean)
                    self.logger.logkv("eval/episode_reward", ep_reward_mean)
                    self.logger.logkv("eval/episode_reward_std", ep_reward_std)
                
                # Log success rate for MetaWorld environments
                if "eval/episode_success" in eval_info:
                    ep_success_mean = np.mean(eval_info["eval/episode_success"]) * 100  # Convert to percentage
                    ep_success_std = np.std(eval_info["eval/episode_success"]) * 100
                    self.logger.logkv("eval/episode_success", ep_success_mean)
                    self.logger.logkv("eval/episode_success_std", ep_success_std)
                
                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)

            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
    
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        episode_success = 0  # Track success for MetaWorld environments

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, info = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1
            
            # Track success for MetaWorld environments
            # Check for MetaWorld environment types
            is_metaworld = False
            try:
                # Check environment class names for MetaWorld indicators
                env_to_check = self.eval_env
                while hasattr(env_to_check, '_wrapped_env') or hasattr(env_to_check, 'wrapped_env') or hasattr(env_to_check, 'env'):
                    if hasattr(env_to_check, '_wrapped_env'):
                        env_to_check = env_to_check._wrapped_env
                    elif hasattr(env_to_check, 'wrapped_env'):
                        env_to_check = env_to_check.wrapped_env
                    elif hasattr(env_to_check, 'env'):
                        env_to_check = env_to_check.env
                    else:
                        break
                
                env_name = str(type(env_to_check).__name__)
                if "metaworld" in env_name.lower() or "sawyer" in env_name.lower() or "box" in env_name.lower():
                    is_metaworld = True
            except:
                # Fallback: check if task name contains metaworld
                if hasattr(self, '_task_name') and "metaworld" in str(getattr(self, '_task_name', '')).lower():
                    is_metaworld = True
            
            if is_metaworld and info and "success" in info:
                episode_success = max(episode_success, info["success"])

            obs = next_obs

            if terminal:
                episode_info = {
                    "episode_reward": episode_reward, 
                    "episode_length": episode_length
                }
                
                # Add success rate for MetaWorld environments
                if is_metaworld:
                    episode_info["episode_success"] = episode_success
                    
                eval_ep_info_buffer.append(episode_info)
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                episode_success = 0
                obs = self.eval_env.reset()
        
        result = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
        
        # Add success rate if available
        if eval_ep_info_buffer and "episode_success" in eval_ep_info_buffer[0]:
            result["eval/episode_success"] = [ep_info["episode_success"] for ep_info in eval_ep_info_buffer]
        
        return result
