import numpy as np
import torch
import torch.nn as nn
import gym

from copy import deepcopy
from typing import Dict, Union, Tuple, Optional
from offlinepbrl.policy import IQLPolicy


class AdversarialIPLIQLPolicy(IQLPolicy):
    """
    Adversarial Inverse Preference Learning IQL (A-IPL-IQL)
    
    Implements adversarial training where the reward model is trained to be pessimistic
    in OOD regions while still respecting preference constraints. The key idea is to
    minimize Q-values on replay data (adversarial objective) while maintaining
    preference consistency (preference objective).
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
        adversarial_weight: float = 0.1,  # New: weight for adversarial loss
        adversarial_type: str = "replay",  # Type of adversarial loss
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
        self.adversarial_weight = adversarial_weight  # New parameter
        self.adversarial_type = adversarial_type  # Type of adversarial loss
        self.reg_replay_weight = reg_replay_weight
        self.actor_replay_weight = actor_replay_weight
        self.value_replay_weight = value_replay_weight
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
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
        
        if self.actor_replay_weight is not None:
            a1, a2, ar = torch.split(actor_loss, split, dim=0)
            actor_loss_fb = (a1.mean() + a2.mean()) / 2
            actor_loss_re = ar.mean()
            actor_loss = (1 - self.actor_replay_weight) * actor_loss_fb + self.actor_replay_weight * actor_loss_re
        else:
            actor_loss = actor_loss.mean()
            
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # compute the critic loss using Inverse Bellman Operator + Adversarial Training
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
        
        # Adversarial loss: minimize Q-values on replay data to encourage pessimism
        # Split Q-values for replay data only
        q1_split = torch.split(q1_pred, split, dim=0)  # Split q1_pred into [pref1, pref2, replay]
        q2_split = torch.split(q2_pred, split, dim=0)  # Split q2_pred into [pref1, pref2, replay]
        q1_pref1, q1_pref2, q1_replay = q1_split[0], q1_split[1], q1_split[2]
        q2_pref1, q2_pref2, q2_replay = q2_split[0], q2_split[1], q2_split[2]
        
        # Compute adversarial loss based on type
        if self.adversarial_type == "replay":
            adversarial_loss = (q1_replay.mean() + q2_replay.mean()) / 2
        elif self.adversarial_type == "replay-pref":
            q_replay_mean = (q1_replay.mean() + q2_replay.mean()) / 2
            q_pref_mean = (q1_pref1.mean() + q1_pref2.mean() + q2_pref1.mean() + q2_pref2.mean()) / 4
            adversarial_loss = q_replay_mean - q_pref_mean
        elif self.adversarial_type == "replay+pref":
            q_replay_mean = (q1_replay.mean() + q2_replay.mean()) / 2
            q_pref_mean = (q1_pref1.mean() + q1_pref2.mean() + q2_pref1.mean() + q2_pref2.mean()) / 4
            adversarial_loss = q_replay_mean + q_pref_mean
        elif self.adversarial_type == "replay+lose-win":
            q_replay_mean = (q1_replay.mean() + q2_replay.mean()) / 2
            # Use labels to determine win/lose trajectories dynamically
            # When label=1, pref2 wins; when label=0, pref1 wins
            # Need to reshape Q values first: [F_B*F_S, 1] -> [F_B, F_S, 1]
            q1_pref1_reshaped = q1_pref1.reshape(F_B, F_S, -1)
            q1_pref2_reshaped = q1_pref2.reshape(F_B, F_S, -1)
            q2_pref1_reshaped = q2_pref1.reshape(F_B, F_S, -1)
            q2_pref2_reshaped = q2_pref2.reshape(F_B, F_S, -1)
            label_mask = labels[0].unsqueeze(-1).unsqueeze(-1)  # Shape: [F_B, 1, 1]
            q1_win = torch.where(label_mask == 1, q1_pref2_reshaped, q1_pref1_reshaped)
            q1_lose = torch.where(label_mask == 0, q1_pref2_reshaped, q1_pref1_reshaped)
            q2_win = torch.where(label_mask == 1, q2_pref2_reshaped, q2_pref1_reshaped)
            q2_lose = torch.where(label_mask == 0, q2_pref2_reshaped, q2_pref1_reshaped)
            q_lose_mean = (q1_lose.mean() + q2_lose.mean()) / 2
            q_win_mean = (q1_win.mean() + q2_win.mean()) / 2
            adversarial_loss = q_replay_mean + q_lose_mean - q_win_mean
        elif self.adversarial_type == "replay-win":
            q_replay_mean = (q1_replay.mean() + q2_replay.mean()) / 2
            # Use labels to determine win trajectory
            # Need to reshape Q values first: [F_B*F_S, 1] -> [F_B, F_S, 1]
            q1_pref1_reshaped = q1_pref1.reshape(F_B, F_S, -1)
            q1_pref2_reshaped = q1_pref2.reshape(F_B, F_S, -1)
            q2_pref1_reshaped = q2_pref1.reshape(F_B, F_S, -1)
            q2_pref2_reshaped = q2_pref2.reshape(F_B, F_S, -1)
            label_mask = labels[0].unsqueeze(-1).unsqueeze(-1)  # Shape: [F_B, 1, 1]
            q1_win = torch.where(label_mask == 1, q1_pref2_reshaped, q1_pref1_reshaped)
            q2_win = torch.where(label_mask == 1, q2_pref2_reshaped, q2_pref1_reshaped)
            q_win_mean = (q1_win.mean() + q2_win.mean()) / 2
            adversarial_loss = q_replay_mean - q_win_mean
        else:
            raise ValueError(f"Unknown adversarial_type: {self.adversarial_type}")

        
        # Regularization loss
        reg_loss_fb = (r1.square().mean() + r2.square().mean()) / 2
        reg_loss_re = rr.square().mean()
        if self.reg_replay_weight is not None:
            reg_loss = (1 - self.reg_replay_weight) * reg_loss_fb + self.reg_replay_weight * reg_loss_re
        else:
            reg_loss = reward.square().mean()
        
        # Total critic loss: preference consistency + adversarial objective + regularization
        critic_loss = pref_loss + self.adversarial_weight * adversarial_loss + self.reward_reg * reg_loss
        
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
            "loss/adversarial": adversarial_loss.item(),  # New metric
            "loss/v": v_loss.item(),
            "loss/reg": reg_loss.item(),
            "misc/reward_value": reward.mean().item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/q1": q1_pred.mean().item(),
            "misc/q2": q2_pred.mean().item(),
            "misc/next_v": next_v.mean().item(),
            "misc/q_replay_mean": (q1_replay.mean() + q2_replay.mean()).item() / 2,  # New metric
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