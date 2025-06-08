import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinepbrl.policy import BasePolicy


class AWACPolicy(BasePolicy):
    """
    Advantage Weighted Actor Critic <Ref: https://arxiv.org/abs/2006.09359>
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
        gamma: float  = 0.99,
        temperature: float = 0.1
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q1_old.eval()
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q2_old.eval()

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._temperature = temperature

    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()

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
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions)
        with torch.no_grad():
            next_actions = self.actor(next_obss).sample()
            next_q1 = self.critic_q1_old(next_obss, next_actions)
            next_q2 = self.critic_q2_old(next_obss, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # update actor
        with torch.no_grad():
            baseline_actions = self.actor(obss).sample()
            baseline_q1 = self.critic_q1_old(obss, baseline_actions)
            baseline_q2 = self.critic_q2_old(obss, baseline_actions)
            baseline_q = torch.min(baseline_q1, baseline_q2)
            
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
            
            advantage = q - baseline_q
            exp_a = torch.exp(advantage / self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        
        dist = self.actor(obss)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "misc/advantage": advantage.mean().item(),
            "misc/q1": q1.mean().item(),
            "misc/q2": q2.mean().item(),
            "misc/next_q": next_q.mean().item(),
        }