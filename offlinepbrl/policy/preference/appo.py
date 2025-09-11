import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy
from typing import Dict, Union, Tuple

from offlinepbrl.policy import BasePolicy


class APPOPolicy(BasePolicy):
    """
    Adversarial Preference Policy Optimization (APPO)
    Based on the original APPO implementation
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
        lam: float = 1e-3,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
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
        self._target_entropy = -np.prod(self.action_space.shape)
        
        self._lam = lam
        self._tau = tau
        self._gamma = gamma
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.exp()
        else:
            self._alpha = alpha

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
        obs = torch.as_tensor(obs, device=next(self.actor.parameters()).device, dtype=torch.float32)
        
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                action, _ = dist.mode()
            else:
                action = dist.sample()
        
        action = np.clip(action.cpu().numpy(), self.action_space.low[0], self.action_space.high[0])
        return action

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # Sample actions from current policy
        sampled_actions_dist = self.actor(obss)
        sampled_actions, raw_sampled_actions = sampled_actions_dist.rsample()
        sampled_log_probs = sampled_actions_dist.log_prob(sampled_actions, raw_sampled_actions).sum(dim=-1, keepdim=True)
        
        # Compute Q values for both sampled and dataset actions
        q1_sampled = self.critic_q1(obss, sampled_actions.detach())
        q2_sampled = self.critic_q2(obss, sampled_actions.detach())
        q1_dataset = self.critic_q1(obss, actions)
        q2_dataset = self.critic_q2(obss, actions)
        
        # Adversarial loss for critics (key part of APPO)
        adv_loss_1 = (q1_sampled - q1_dataset).mean()
        adv_loss_2 = (q2_sampled - q2_dataset).mean()
        
        # Standard TD loss for critics
        with torch.no_grad():
            next_actions_dist = self.actor(next_obss)
            next_actions, next_raw_actions = next_actions_dist.rsample()
            next_log_probs = next_actions_dist.log_prob(next_actions, next_raw_actions).sum(dim=-1, keepdim=True)
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * (1 - terminals) * next_v
        
        td_loss_1 = ((q1_dataset - target_q).pow(2)).mean()
        td_loss_2 = ((q2_dataset - target_q).pow(2)).mean()
        
        # Combined critic loss: adversarial + TD
        critic_q1_loss = self._lam * adv_loss_1 + td_loss_1
        critic_q2_loss = self._lam * adv_loss_2 + td_loss_2

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # Update V network
        with torch.no_grad():
            target_v = torch.min(
                self.critic_q1_old(obss, sampled_actions.detach()), 
                self.critic_q2_old(obss, sampled_actions.detach())
            )
        v = self.critic_v(obss)
        critic_v_loss = ((v - target_v).pow(2)).mean()
        
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # Update actor
        # Randomly choose which Q network to use for actor update (as in original APPO)
        if np.random.random() < 0.5:
            q_sampled = self.critic_q1(obss, sampled_actions)
        else:
            q_sampled = self.critic_q2(obss, sampled_actions)
        
        actor_loss = (self._alpha * sampled_log_probs - q_sampled).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Update alpha if auto_alpha is enabled
        if self._is_auto_alpha:
            alpha_loss = -(self._log_alpha * (sampled_log_probs + self._target_entropy).detach()).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "loss/v": critic_v_loss.item(),
            "loss/adv": (adv_loss_1 + adv_loss_2).item(),
            "misc/q1": q1_dataset.mean().item(),
            "misc/q2": q2_dataset.mean().item(),
            "misc/v": v.mean().item(),
            "misc/alpha": self._alpha.item() if self._is_auto_alpha else self._alpha,
        }
        
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            
        return result
