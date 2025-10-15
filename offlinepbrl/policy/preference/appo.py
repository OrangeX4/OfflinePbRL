import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import random

from copy import deepcopy
from typing import Dict, Union, Tuple, Optional
from offlinepbrl.policy import BasePolicy


class APPOPolicy(BasePolicy):
    """
    Adversarial Preference-based Policy Optimization (APPO)
    <Ref: Adversarial Policy Optimization for Offline Preference-Based Reinforcement Learning>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float = 0.99,
        lam: float = 0.03,  # adversarial loss coefficient
        alpha: float = 0.2,  # entropy regularization
        auto_alpha: bool = True,
        alpha_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q1_old.eval()
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q2_old.eval()
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._lam = lam  # adversarial loss coefficient
        
        # Entropy regularization
        self._auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy is None:
                self._target_entropy = -np.prod(action_space.shape)
            else:
                self._target_entropy = target_entropy
            self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
            self._alpha = self._log_alpha.exp().detach()
        else:
            self._alpha = alpha
        
        self._device = device
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        
        obs = torch.tensor(obs, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            dist = self.actor(obs)
            # Support both NormalWrapper and TanhNormalWrapper
            if deterministic:
                mode_out = dist.mode()
                action = mode_out[0] if isinstance(mode_out, (tuple, list)) else mode_out
            else:
                sample_out = dist.rsample()
                action = sample_out[0] if isinstance(sample_out, (tuple, list)) else sample_out
            action = action.cpu().numpy()
        
        # Clip to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def _get_action_and_log_prob(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability; prefer TanhDiagGaussian's corrected log_prob when available."""
        dist = self.actor(obs)
        if deterministic:
            mode_out = dist.mode()
            if isinstance(mode_out, (tuple, list)):
                action, raw_action = mode_out
                log_prob = dist.log_prob(action, raw_action)
            else:
                # Fallback: unsquashed distribution
                action = mode_out
                log_prob = dist.log_prob(action)
        else:
            sample_out = dist.rsample()
            if isinstance(sample_out, (tuple, list)):
                action, raw_action = sample_out
                log_prob = dist.log_prob(action, raw_action)
            else:
                action = sample_out
                log_prob = dist.log_prob(action)
        return action, log_prob
    
    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        # Handle different input formats
        if isinstance(batch, dict) and ("replay" in batch or "traj" in batch):
            # Called with combined batch from trainer
            if "replay" in batch:
                replay_batch = batch["replay"]
            else:
                # Direct batch without wrapper structure
                replay_batch = {k: v for k, v in batch.items() if k != "traj"}
            
            if "traj" in batch:
                traj_batch = batch["traj"]
            else:
                raise ValueError("APPO requires trajectory batch, but 'traj' not found in batch")
        else:
            # Direct call format (shouldn't happen in normal training)
            raise ValueError("APPOPolicy expects batch dict with 'replay' and 'traj' keys")
            
        obss, actions, next_obss, rewards, terminals = replay_batch["observations"], replay_batch["actions"], \
            replay_batch["next_observations"], replay_batch["rewards"], replay_batch["terminals"]
        
        # Trajectory batch for regularization
        traj_obs, traj_actions, traj_next_obs, traj_rewards, traj_terminals = \
            traj_batch["observations"], traj_batch["actions"], traj_batch["next_observations"], \
            traj_batch["rewards"], traj_batch["terminals"]
        
        log_dict = {}
        
        # Calculate trajectory regularization loss
        # Prefer metadata from traj_batch to avoid hard-coded sizes
        if "segment_size" in traj_batch and "pair_batch_size" in traj_batch:
            segment_size = int(traj_batch["segment_size"])
            half_size = int(traj_batch["pair_batch_size"])
        else:
            segment_size = traj_obs.size(0) // (2 * 16)  # fallback
            half_size = 16
        
        # Compute trajectory-level TD targets for regularization
        with torch.no_grad():
            next_v_traj = self.critic_v(traj_next_obs).flatten()
            # Mask terminal transitions to avoid bootstrapping beyond episode ends
            terminals_traj = traj_terminals.flatten()
            target_traj = traj_rewards.flatten() + self._gamma * (1.0 - terminals_traj) * next_v_traj
            # Sum over segments to get trajectory returns
            target_traj = torch.sum(target_traj.view(-1, segment_size), dim=-1)

        # Get Q values for trajectory segments
        q1_traj = self.critic_q1(traj_obs, traj_actions).flatten()
        q2_traj = self.critic_q2(traj_obs, traj_actions).flatten()
        q1_traj_sum = torch.sum(q1_traj.view(-1, segment_size), dim=-1)
        q2_traj_sum = torch.sum(q2_traj.view(-1, segment_size), dim=-1)
        
        # Trajectory pair L1 loss (between first half and second half of trajectory pairs)
        traj_reg_loss_1 = (q1_traj_sum[:half_size] - target_traj[:half_size] - 
                           q1_traj_sum[half_size:] + target_traj[half_size:]).abs().mean() / segment_size
        traj_reg_loss_2 = (q2_traj_sum[:half_size] - target_traj[:half_size] - 
                           q2_traj_sum[half_size:] + target_traj[half_size:]).abs().mean() / segment_size

        # Get current policy actions for adversarial loss
        current_actions, log_probs = self._get_action_and_log_prob(obss)
        
        # Calculate adversarial loss
        q1_current = self.critic_q1(obss, current_actions.detach()).flatten()
        q2_current = self.critic_q2(obss, current_actions.detach()).flatten()
        q1_data = self.critic_q1(obss, actions).flatten()
        q2_data = self.critic_q2(obss, actions).flatten()
        
        adv_loss_1 = (q1_current - q1_data).mean()
        adv_loss_2 = (q2_current - q2_data).mean()
        
        # Combined critic losses
        critic1_loss = self._lam * adv_loss_1 + traj_reg_loss_1
        critic2_loss = self._lam * adv_loss_2 + traj_reg_loss_2

        # Update Q-networks
        self.critic_q1_optim.zero_grad()
        critic1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic2_loss.backward()
        self.critic_q2_optim.step()

        # Update V-network (similar to IQL expectile regression, but simpler MSE with min Q)
        with torch.no_grad():
            q_target = torch.minimum(
                self.critic_q1_old(obss, current_actions.detach()),
                self.critic_q2_old(obss, current_actions.detach())
            ).flatten()
        
        v_current = self.critic_v(obss).flatten()
        v_loss = F.mse_loss(v_current, q_target)
        
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()

        # Sync target networks
        self._sync_weight()

        # Update actor policy
        # Randomly choose which Q-network to use for policy update (as in local implementation)
        idx = random.choice([0, 1])
        if idx == 0:
            q_values = self.critic_q1(obss, current_actions).flatten()
        else:
            q_values = self.critic_q2(obss, current_actions).flatten()
        
        actor_loss = (self._alpha * log_probs.flatten() - q_values).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update alpha (entropy regularization) if auto_alpha is enabled
        if self._auto_alpha:
            alpha_loss = -(self._log_alpha * (log_probs.detach() + self._target_entropy)).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp().detach()
            log_dict["loss/alpha"] = alpha_loss.item()
            log_dict["misc/alpha"] = self._alpha.item()

        # Logging
        log_dict.update({
            "loss/actor": actor_loss.item(),
            "loss/q1": critic1_loss.item(),
            "loss/q2": critic2_loss.item(),
            "loss/v": v_loss.item(),
            "misc/q1": q1_data.mean().item(),
            "misc/q2": q2_data.mean().item(),
            "misc/v": v_current.mean().item(),
            "misc/log_probs": log_probs.mean().item(),
            "misc/adv_loss": (adv_loss_1.item() + adv_loss_2.item()),
            "misc/traj_reg_loss": (traj_reg_loss_1.item() + traj_reg_loss_2.item()),
        })

        return log_dict