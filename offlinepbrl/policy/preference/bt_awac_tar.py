import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple, Optional
from offlinepbrl.policy import AWACPolicy
from offlinepbrl.modules.reward_module import BaseRewardModel


class BTAWACTARPolicy(AWACPolicy):
    """
    Bradley-Terry AWAC with Trajectory Adherence Regularization (TAR)
    
    This combines:
    1. Bradley-Terry preference learning for reward model
    2. AWAC policy learning
    3. TAR trajectory-level regularization
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        reward_model: BaseRewardModel,
        reward_model_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        temperature: float = 3.0,
        reward_reg: float = 0.0,
        rm_stop_epoch: Optional[int] = None,
        policy_start_epoch: Optional[int] = None,
        tar_coef: float = 1.0,
        tar_clip_max: float = 100.0,
    ) -> None:
        super().__init__(
            actor=actor,
            critic_q1=critic_q1,
            critic_q2=critic_q2,
            actor_optim=actor_optim,
            critic_q1_optim=critic_q1_optim,
            critic_q2_optim=critic_q2_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            temperature=temperature
        )
        self.reward_model = reward_model
        self.reward_model_optim = reward_model_optim
        self.reward_reg = reward_reg
        self.rm_stop_epoch = rm_stop_epoch
        self.policy_start_epoch = policy_start_epoch
        self.tar_coef = tar_coef
        self.tar_clip_max = tar_clip_max
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def compute_tar_loss(
        self, 
        obs_traj: torch.Tensor, 
        action_traj: torch.Tensor, 
        traj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Trajectory Adherence Regularization (TAR) loss.
        
        TAR formula: L_TAR = E_τ [ (Σ_t ||π(s_t) - a_t||)² ]
        where ||·|| is L2 norm (with sqrt)
        
        Args:
            obs_traj: [batch_size, max_len, obs_dim]
            action_traj: [batch_size, max_len, action_dim]
            traj_mask: [batch_size, max_len] - 1.0 for valid steps, 0.0 for padding
            
        Returns:
            tar_loss: scalar
            metrics: dict with auxiliary metrics
        """
        batch_size, max_len = obs_traj.shape[:2]
        
        # Flatten to [batch_size * max_len, ...]
        obs_flat = obs_traj.reshape(-1, obs_traj.shape[-1])
        action_flat = action_traj.reshape(-1, action_traj.shape[-1])
        mask_flat = traj_mask.reshape(-1)
        
        # Get policy actions (need gradients for backprop)
        dist = self.actor(obs_flat)
        if hasattr(dist, 'mode'):
            policy_actions = dist.mode()
        else:
            policy_actions = dist.mean
        
        # Compute per-step L2 distances (with sqrt)
        # ||π(s_t) - a_t|| = sqrt(sum_i (π_i - a_i)^2)
        step_distances = torch.norm(policy_actions - action_flat, p=2, dim=-1)  # [batch_size * max_len]
        
        # Clip per-step distances
        step_distances = torch.clamp(step_distances, max=self.tar_clip_max)
        
        # Apply mask
        step_distances = step_distances * mask_flat
        
        # Reshape back to trajectories
        step_distances = step_distances.reshape(batch_size, max_len)
        
        # Compute trajectory lengths for normalization
        traj_lengths = traj_mask.sum(dim=1)  # [batch_size]
        
        # Compute trajectory-level cumulative distance: D(π, τ) = Σ_t ||π(s_t) - a_t||
        traj_cumulative_distance = step_distances.sum(dim=1)  # [batch_size]
        
        # Normalize by trajectory length to handle variable-length trajectories
        normalized_distance = traj_cumulative_distance / (traj_lengths + 1e-8)
        
        # TAR loss: (D(π, τ)/T)^2 averaged over trajectories
        tar_loss = (normalized_distance.pow(2)).mean()
        
        # Compute metrics (use unnormalized for interpretability)
        metrics = {
            "tar/loss": tar_loss.item(),
            "tar/avg_cumulative_distance": traj_cumulative_distance.mean().item(),
            "tar/avg_normalized_distance": normalized_distance.mean().item(),
            "tar/avg_step_distance": (step_distances.sum() / mask_flat.sum()).item(),
            "tar/max_cumulative_distance": traj_cumulative_distance.max().item(),
        }
        
        return tar_loss, metrics

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        """
        Learn with BT preference learning, AWAC policy, and TAR.
        
        batch should contain:
            - "replay": transition-level data for AWAC
            - "pref": preference data for BT
            - "trajectory": trajectory-level data for TAR (optional)
        """
        replay_batch = batch["replay"]
        pref_batch = batch.get("pref", None)
        traj_batch = batch.get("trajectory", None)
        
        # Determine learning phases
        should_train_rm = True
        if epoch is not None and self.rm_stop_epoch is not None:
            should_train_rm = epoch < self.rm_stop_epoch
        
        should_train_policy = True
        if epoch is not None and self.policy_start_epoch is not None:
            should_train_policy = epoch >= self.policy_start_epoch
        
        metrics = {}
        
        # 1. Bradley-Terry preference learning
        if should_train_rm and pref_batch is not None:
            F_B, F_S = pref_batch["obs_1"].shape[0:2]
            F_S -= 1
            
            pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
            pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
            pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
            pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
            
            reward_1 = self.reward_model(pref_obs_1, pref_action_1).reshape(F_B, F_S)
            reward_2 = self.reward_model(pref_obs_2, pref_action_2).reshape(F_B, F_S)
            
            logits = reward_2.sum(dim=-1) - reward_1.sum(dim=-1)
            labels = pref_batch["label"][:, 1].float()
            pref_loss = self.reward_criterion(logits, labels).mean()
            
            # Regularization
            reg_loss = (reward_1.square().mean() + reward_2.square().mean()) / 2
            reward_model_loss = pref_loss + self.reward_reg * reg_loss
            
            self.reward_model_optim.zero_grad()
            reward_model_loss.backward()
            self.reward_model_optim.step()
            
            with torch.no_grad():
                reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
            
            metrics.update({
                "loss/preference": pref_loss.item(),
                "loss/reward_model": reward_model_loss.item(),
                "loss/reg": reg_loss.item(),
                "misc/reward_acc": reward_accuracy.item(),
            })
        
        # 2. AWAC policy learning with learned rewards
        if should_train_policy:
            # Replace rewards with learned rewards
            with torch.no_grad():
                replay_rewards = self.reward_model(replay_batch["observations"], replay_batch["actions"])
                replay_batch["rewards"] = replay_rewards
            
            # Standard AWAC update
            awac_metrics = super().learn(replay_batch, epoch, step)
            metrics.update(awac_metrics)
        
        # 3. TAR regularization
        if should_train_policy and traj_batch is not None and self.tar_coef > 0:
            obs_traj = traj_batch["observations"]
            action_traj = traj_batch["actions"]
            traj_mask = traj_batch["trajectory_mask"]
            
            tar_loss, tar_metrics = self.compute_tar_loss(obs_traj, action_traj, traj_mask)
            
            # Apply TAR to actor
            total_tar_loss = self.tar_coef * tar_loss
            
            self.actor_optim.zero_grad()
            total_tar_loss.backward()
            self.actor_optim.step()
            
            metrics.update(tar_metrics)
            metrics["loss/tar_weighted"] = total_tar_loss.item()
        
        return metrics
