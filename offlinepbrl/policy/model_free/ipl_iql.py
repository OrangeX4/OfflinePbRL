import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple, Optional
from offlinepbrl.policy import IQLPolicy


class IPLIQLPolicy(IQLPolicy):
    """
    Inverse Preference Learning IQL
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
        expectile: float = 0.8,
        temperature: float = 0.1,
        reward_reg: float = 0.5,
        reg_replay_weight: Optional[float] = None,
        actor_replay_weight: Optional[float] = None,
        value_replay_weight: Optional[float] = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic_q1=critic_q1,
            critic_q2=critic_q2,
            critic_v=critic_v,
            actor_optim=actor_optim,
            critic_q1_optim=critic_q1_optim,
            critic_q2_optim=critic_q2_optim,
            critic_v_optim=critic_v_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            expectile=expectile,
            temperature=temperature
        )
        self.reward_reg = reward_reg
        self.reg_replay_weight = reg_replay_weight
        self.actor_replay_weight = actor_replay_weight
        self.value_replay_weight = value_replay_weight
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def learn(self, batch: Dict) -> Dict[str, float]:
        replay_batch, pref_batch = batch["replay"], batch["pref"]

        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        R_B = replay_batch["observations"].shape[0]
        split = [F_B*F_S, F_B*F_S, R_B]

        # Concatenate preference and replay data
        obs = torch.concat([
            pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1),
            pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1),
            replay_batch["observations"],
        ], dim=0)
        
        next_obs = torch.concat([
            pref_batch["obs_1"][:, 1:].reshape(F_B*F_S, -1),
            pref_batch["obs_2"][:, 1:].reshape(F_B*F_S, -1),
            replay_batch["next_observations"],
        ], dim=0)
        
        action = torch.concat([
            pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1),
            pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1),
            replay_batch["actions"],
        ], dim=0)
        
        terminal = torch.concat([
            pref_batch["terminal_1"][:, :-1].reshape(F_B*F_S, -1),
            pref_batch["terminal_2"][:, :-1].reshape(F_B*F_S, -1),
            replay_batch["terminals"],
        ], dim=0)

        # compute value loss
        with torch.no_grad():
            q1_old, q2_old = self.critic_q1_old(obs, action), self.critic_q2_old(obs, action)
            q_old = torch.min(q1_old, q2_old)
        v_loss = self._expectile_regression(q_old - self.critic_v(obs))
        v1, v2, vr = torch.split(v_loss, split, dim=0)
        v_loss_fb = (v1.mean() + v2.mean()) / 2
        v_loss_re = vr.mean()
        v_loss = (1 - self.value_replay_weight) * v_loss_fb + self.value_replay_weight * v_loss_re
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()

        # compute actor loss
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obs, action), self.critic_q2_old(obs, action)
            q = torch.min(q1, q2)
            v = self.critic_v(obs)
            exp_a = torch.exp((q - v) * self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        dist = self.actor(obs)
        log_probs = dist.log_prob(action)
        actor_loss = -(exp_a * log_probs)
        a1, a2, ar = torch.split(actor_loss, split, dim=0)
        actor_loss_fb = (a1.mean() + a2.mean()) / 2
        actor_loss_re = ar.mean()
        actor_loss = (1 - self.actor_replay_weight) * actor_loss_fb + self.actor_replay_weight * actor_loss_re
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # compute the critic loss using Inverse Bellman Operator
        q1_pred = self.critic_q1(obs.detach(), action)
        q2_pred = self.critic_q2(obs.detach(), action)
        with torch.no_grad():
            next_v = self.critic_v(next_obs)
        # Inverse Bellman: reward = Q - γ * V_next
        reward1 = q1_pred - (1 - terminal) * self._gamma * next_v
        reward2 = q2_pred - (1 - terminal) * self._gamma * next_v
        # Stack along first dimension to create ensemble-like structure
        reward = torch.stack([reward1, reward2], dim=0)  # Shape: [2, total_size, 1]
        reward = reward.squeeze(-1)  # Shape: [2, total_size]
        
        # Split rewards for preference learning
        r1, r2, rr = torch.split(reward, split, dim=1)
        E = r1.shape[0]
        r1, r2 = r1.reshape(E, F_B, F_S), r2.reshape(E, F_B, F_S)
        
        # Bradley-Terry model for preferences
        logits = r2.sum(dim=-1) - r1.sum(dim=-1)  # Shape: [E, F_B]
        labels = pref_batch["label"][:, 1].float().unsqueeze(0).expand(E, -1)  # Shape: [E, F_B]
        pref_loss = self.reward_criterion(logits, labels).mean()
        
        # Regularization loss
        reg_loss_fb = (r1.square().mean() + r2.square().mean()) / 2
        reg_loss_re = rr.square().mean()
        reg_loss = (1 - self.reg_replay_weight) * reg_loss_fb + self.reg_replay_weight * reg_loss_re
        
        critic_loss = pref_loss + self.reward_reg * reg_loss
        
        self.critic_q1_optim.zero_grad()
        self.critic_q2_optim.zero_grad()
        critic_loss.backward()
        self.critic_q1_optim.step()
        self.critic_q2_optim.step()

        self._sync_weight()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        metrics = {
            "loss/actor": actor_loss.item(),
            "loss/preference": pref_loss.item(),
            "loss/v": v_loss.item(),
            "loss/reg": reg_loss.item(),
            "misc/reward_value": reward.mean().item(),
            "misc/reward_acc": reward_accuracy.item()
        }
        
        if self.actor_replay_weight is not None:
            metrics.update({
                "detail/actor_loss_fb": actor_loss_fb.item(),
                "detail/actor_loss_re": actor_loss_re.item(),
            })
        if self.value_replay_weight is not None:
            metrics.update({
                "detail/v_loss_fb": v_loss_fb.item(),
                "detail/v_loss_re": v_loss_re.item(),
            })
        if self.reg_replay_weight is not None:
            metrics.update({
                "detail/reg_loss_fb": reg_loss_fb.item(),
                "detail/reg_loss_re": reg_loss_re.item()
            })

        return metrics