import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy
from typing import Dict, Union, Tuple

from offlinepbrl.policy import BasePolicy


class APPOPolicy(BasePolicy):
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
        
        self._lam = lam
        self._tau = tau
        self._gamma = gamma
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._log_alpha, self._alpha_optim = alpha
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

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
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

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions)
        with torch.no_grad():
            next_actions_dist = self.actor(next_obss)
            next_actions = next_actions_dist.sample()
            log_probs = next_actions_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            next_q = torch.min(
                self.critic_q1_old(next_obss, next_actions), self.critic_q2_old(next_obss, next_actions)
            )
            target_q = rewards + self._gamma * (1 - terminals) * (next_q - self._alpha * log_probs)
        
        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # update actor
        # non-adversarial loss
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
        
        # adversarial loss
        sampled_actions_dist = self.actor(obss)
        sampled_actions = sampled_actions_dist.rsample()
        q_sampled = torch.min(self.critic_q1(obss, sampled_actions), self.critic_q2(obss, sampled_actions))
        
        actor_loss = (self._alpha * sampled_actions_dist.log_prob(sampled_actions).sum(dim=-1) - q_sampled).mean()
        
        l2_loss = nn.MSELoss()
        behavior_actions_dist = self.actor(obss)
        behavior_actions = behavior_actions_dist.sample()
        
        raw_actions = behavior_actions_dist.log_prob(actions).sum(dim=-1)
        raw_sampled_actions = behavior_actions_dist.log_prob(behavior_actions).sum(dim=-1)
        
        actor_loss += self._lam * l2_loss(raw_actions, raw_sampled_actions)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        if self._is_auto_alpha:
            # 计算目标熵，通常为 -action_dim
            target_entropy = -float(self.action_space.shape[0])
            log_probs = sampled_actions_dist.log_prob(sampled_actions).sum(dim=-1) + target_entropy
            alpha_loss = -(self._log_alpha * log_probs.detach()).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "misc/q1": q1.mean().item(),
            "misc/q2": q2.mean().item(),
            "misc/alpha": self._alpha.item() if self._is_auto_alpha else self._alpha,
        }
        
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            
        return result
