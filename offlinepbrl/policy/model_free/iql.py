import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinepbrl.policy import BasePolicy


class IQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
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
        gamma: float  = 0.99,
        expectile: float = 0.8,
        temperature: float = 0.1
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
        self._expectile = expectile
        self._temperature = temperature

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
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                action = dist.mode().cpu().numpy()
            else:
                action = dist.sample().cpu().numpy()
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action
    
    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric L2 loss for expectile regression.
        When diff > 0 (Q > V): weight with expectile
        When diff < 0 (Q < V): weight with (1 - expectile)
        """
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff ** 2)
    
    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # Step 1: Update V network
        with torch.no_grad():
            # Use target Q for V update
            target_q1 = self.critic_q1_old(obss, actions)
            target_q2 = self.critic_q2_old(obss, actions)
            target_q = torch.min(target_q1, target_q2)
        
        # Compute V and advantage
        v = self.critic_v(obss)
        adv = target_q - v
        
        # Expectile regression loss
        v_loss = self._expectile_regression(adv).mean()
        
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()

        # Step 2: Update Q networks (separately to avoid correlation)
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * (1 - terminals) * next_v
        
        # Update Q1
        q1 = self.critic_q1(obss, actions)
        q1_loss = ((q1 - target_q) ** 2).mean()
        
        self.critic_q1_optim.zero_grad()
        q1_loss.backward()
        self.critic_q1_optim.step()
        
        # Update Q2
        q2 = self.critic_q2(obss, actions)
        q2_loss = ((q2 - target_q) ** 2).mean()
        
        self.critic_q2_optim.zero_grad()
        q2_loss.backward()
        self.critic_q2_optim.step()

        # Step 3: Update target networks
        self._sync_weight()

        # Step 4: Update actor with AWR
        with torch.no_grad():
            # Recompute Q and V after updates
            q1 = self.critic_q1_old(obss, actions)
            q2 = self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
            v = self.critic_v(obss)
            
            # Compute advantage and AWR weight
            adv_actor = q - v
            # Apply temperature (beta) scaling
            exp_adv = torch.exp(self._temperature * adv_actor)
            # Clamp to prevent numerical issues
            exp_adv = torch.clamp(exp_adv, max=100.0)
        
        # Get log probability
        dist = self.actor(obss)
        log_prob = dist.log_prob(actions)
        
        # Handle multi-dimensional actions
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # AWR loss: maximize weighted log probability
        actor_loss = -(exp_adv * log_prob).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": q1_loss.item(),
            "loss/q2": q2_loss.item(),
            "loss/v": v_loss.item(),
            "misc/q1": q1.mean().item(),
            "misc/q2": q2.mean().item(),
            "misc/v": v.mean().item(),
            "misc/next_v": next_v.mean().item(),
            "misc/adv_mean": adv.mean().item(),
            "misc/adv_std": adv.std().item(),
            "misc/adv_actor_mean": adv_actor.mean().item(),
            "misc/adv_actor_std": adv_actor.std().item(),
            "misc/exp_adv_mean": exp_adv.mean().item(),
            "misc/exp_adv_max": exp_adv.max().item(),
        }