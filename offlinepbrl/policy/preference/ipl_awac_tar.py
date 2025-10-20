import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple, Optional
from offlinepbrl.policy.preference.ipl_awac import IPLAWACPolicy


class IPLAWACTARPolicy(IPLAWACPolicy):
    """
    Inverse Preference Learning AWAC with Trajectory Adherence Regularization (TAR)
    
    TAR adds a trajectory-level regularization term that penalizes deviations from
    dataset trajectories using a quadratic penalty: (sum of per-step deviations)^2.
    This encourages the policy to maintain trajectory coherence and limits stitching.
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
        tau: float = 0.005,
        gamma: float = 0.99,
        temperature: float = 0.1,
        reward_reg: float = 0.5,
        reg_replay_weight: Optional[float] = None,
        actor_replay_weight: Optional[float] = None,
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
            temperature=temperature,
            reward_reg=reward_reg,
            reg_replay_weight=reg_replay_weight,
            actor_replay_weight=actor_replay_weight,
        )
        self.tar_coef = tar_coef
        self.tar_clip_max = tar_clip_max

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
        
        # Clip per-step distances to prevent outliers
        step_distances = torch.clamp(step_distances, max=self.tar_clip_max)
        
        # Apply mask
        step_distances = step_distances * mask_flat  # [batch_size * max_len]
        
        # Reshape back to trajectories
        step_distances = step_distances.reshape(batch_size, max_len)  # [batch_size, max_len]
        
        # Compute trajectory lengths for normalization
        traj_lengths = traj_mask.sum(dim=1)  # [batch_size]
        
        # Compute trajectory-level cumulative distance
        # D(π, τ) = Σ_t ||π(s_t) - a_t||
        traj_cumulative_distance = step_distances.sum(dim=1)  # [batch_size]
        
        # Normalize by trajectory length to handle variable-length trajectories
        # This makes TAR loss more stable across different trajectory lengths
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
        Learn with both IPL and TAR.
        
        batch should contain:
            - "replay": transition-level data for IPL critic learning
            - "pref": preference data for IPL
            - "trajectory": trajectory-level data for TAR (optional, can use replay if not provided)
        """
        # First do standard IPL-AWAC learning
        metrics = super().learn(batch, epoch, step)
        
        # Now add TAR regularization to actor
        # Check if we have trajectory data, otherwise skip TAR for this step
        if "trajectory" in batch:
            traj_batch = batch["trajectory"]
            
            # Extract trajectory data
            obs_traj = traj_batch["observations"]
            action_traj = traj_batch["actions"]
            traj_mask = traj_batch["trajectory_mask"]
            
            # Compute TAR loss
            tar_loss, tar_metrics = self.compute_tar_loss(obs_traj, action_traj, traj_mask)
            
            # Add TAR to actor loss and do another backward pass
            # Note: We need to recompute actor loss or just add TAR separately
            # For simplicity, we do a separate update step with only TAR
            total_tar_loss = self.tar_coef * tar_loss
            
            self.actor_optim.zero_grad()
            total_tar_loss.backward()
            self.actor_optim.step()
            
            # Update metrics
            metrics.update(tar_metrics)
            metrics["loss/tar_weighted"] = total_tar_loss.item()
        
        return metrics
