import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinepbrl.dynamics import BaseDynamics
from offlinepbrl.utils.scaler import StandardScaler
from offlinepbrl.utils.logger import Logger


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)


def gaussian_cdf(x):
    return 0.5 * (1 + torch.special.erf(x / np.sqrt(2)))

class EnsemblePreferenceDynamics(EnsembleDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim, scaler, terminal_fn, penalty_coef, uncertainty_mode)

    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info

    def format_samples_for_training(self, dataset: Dict, rlhf_dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
        # offline dynamics dataset
        obss = dataset["observations"]
        actions = dataset["actions"]
        next_obss = dataset["next_observations"]
        delta_obss = next_obss - obss
        dynamics_inputs = np.concatenate((obss, actions), axis=-1)
        dynamics_targets = delta_obss

        # rlhf dataset
        obss_1 = rlhf_dataset["observations"]
        actions_1 = rlhf_dataset["actions"]
        obss_2 = rlhf_dataset["observations_2"]
        actions_2 = rlhf_dataset["actions_2"]
        rlhf_inputs_1 = np.concatenate((obss_1, actions_1), axis=-1)
        rlhf_inputs_2 = np.concatenate((obss_2, actions_2), axis=-1)
        rlhf_labels = rlhf_dataset["labels"]
        return dynamics_inputs, dynamics_targets, rlhf_inputs_1, rlhf_inputs_2, rlhf_labels

    def train(
        self,
        dataset: Dict,
        rlhf_dataset: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        dynamics_batch_size: int = 256,
        rlhf_batch_size: int = 32,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        dynamics_inputs, dynamics_targets, rlhf_inputs_1, rlhf_inputs_2, rlhf_labels = self.format_samples_for_training(dataset, rlhf_dataset)

        # offline dynamics dataset
        dynamics_data_size = dynamics_inputs.shape[0]
        dynamics_holdout_size = min(int(dynamics_data_size * holdout_ratio), 1000)
        dynamics_train_size = dynamics_data_size - dynamics_holdout_size
        dynamics_train_splits, dynamics_holdout_splits = torch.utils.data.random_split(range(dynamics_data_size), (dynamics_train_size, dynamics_holdout_size))
        dynamics_train_inputs, dynamics_train_targets = dynamics_inputs[dynamics_train_splits.indices], dynamics_targets[dynamics_train_splits.indices]
        dynamics_holdout_inputs, dynamics_holdout_targets = dynamics_inputs[dynamics_holdout_splits.indices], dynamics_targets[dynamics_holdout_splits.indices]

        self.scaler.fit(dynamics_train_inputs)
        dynamics_train_inputs = self.scaler.transform(dynamics_train_inputs)
        dynamics_holdout_inputs = self.scaler.transform(dynamics_holdout_inputs)

        dynamics_data_idxes = np.random.randint(dynamics_train_size, size=[self.model.num_ensemble, dynamics_train_size])

        # rlhf dataset
        rlhf_data_size = rlhf_inputs_1.shape[0]
        rlhf_holdout_size = min(int(rlhf_data_size * holdout_ratio), 100)
        rlhf_train_size = rlhf_data_size - rlhf_holdout_size
        rlhf_train_splits, rlhf_holdout_splits = torch.utils.data.random_split(range(rlhf_data_size), (rlhf_train_size, rlhf_holdout_size))
        rlhf_train_inputs_1, rlhf_train_inputs_2, rlhf_train_labels = rlhf_inputs_1[rlhf_train_splits.indices], rlhf_inputs_2[rlhf_train_splits.indices], rlhf_labels[rlhf_train_splits.indices]
        rlhf_holdout_inputs_1, rlhf_holdout_inputs_2, rlhf_holdout_labels = rlhf_inputs_1[rlhf_holdout_splits.indices], rlhf_inputs_2[rlhf_holdout_splits.indices], rlhf_labels[rlhf_holdout_splits.indices]

        rlhf_train_inputs_1 = self.scaler.transform(rlhf_train_inputs_1)
        rlhf_train_inputs_2 = self.scaler.transform(rlhf_train_inputs_2)
        rlhf_holdout_inputs_1 = self.scaler.transform(rlhf_holdout_inputs_1)
        rlhf_holdout_inputs_2 = self.scaler.transform(rlhf_holdout_inputs_2)

        rlhf_data_idxes = np.random.randint(rlhf_train_size, size=[self.model.num_ensemble, rlhf_train_size])

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            dynamics_train_loss = self.learn_dynamics(
                dynamics_train_inputs[dynamics_data_idxes],
                dynamics_train_targets[dynamics_data_idxes],
                dynamics_batch_size,
                logvar_loss_coef
            )
            rlhf_train_loss = self.learn_rlhf(
                rlhf_train_inputs_1[rlhf_data_idxes],
                rlhf_train_inputs_2[rlhf_data_idxes],
                rlhf_train_labels[rlhf_data_idxes],
                rlhf_batch_size,
                logvar_loss_coef,
            )
            new_holdout_losses, info = self.validate(
                dynamics_holdout_inputs,
                dynamics_holdout_targets,
                rlhf_holdout_inputs_1,
                rlhf_holdout_inputs_2,
                rlhf_holdout_labels,
            )
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", dynamics_train_loss)
            logger.logkv("loss/rlhf_train_loss", rlhf_train_loss)
            logger.logkv("info/holdout_loss", holdout_loss)
            logger.logkv("info/rlhf_accuracy", info["accuracies"].mean().item())
            logger.logkv("info/reward_mean", info["rewards"].mean().item())
            logger.logkv("info/reward_std", info["rewards"].std().item())
            logger.logkv("info/reward_win_mean", info["rewards_win"].mean().item())
            logger.logkv("info/reward_win_std", info["rewards_win"].std().item())
            logger.logkv("info/reward_lose_mean", info["rewards_lose"].mean().item())
            logger.logkv("info/reward_lose_std", info["rewards_lose"].std().item())
            logger.logkv("info/reward_mean_diff", torch.abs(info["rewards_win"].mean() - info["rewards_lose"].mean()).item())
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            dynamics_data_idxes = shuffle_rows(dynamics_data_idxes)
            rlhf_data_idxes = shuffle_rows(rlhf_data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))

    def learn_dynamics(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            mean, logvar = mean[..., :-1], logvar[..., :-1]
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar[:-1].sum() - logvar_loss_coef * self.model.min_logvar[:-1].sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)

    def learn_rlhf(
        self,
        inputs_1: np.ndarray,
        inputs_2: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        logvar_loss_coef: float = 0.01,
    ) -> float:
        self.model.train()
        train_size = inputs_1.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_1_batch = inputs_1[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            inputs_2_batch = inputs_2[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            labels_batch = labels[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            labels_batch = torch.as_tensor(labels_batch).to(self.model.device)
            
            N, P, M = inputs_1_batch.shape[0], inputs_1_batch.shape[1:-1], inputs_1_batch.shape[-1]
            mean_1, logvar_1 = self.model(inputs_1_batch.reshape(N, -1, M))
            reward_mean_1, logvar_1 = mean_1.reshape(N, *P, -1)[..., -1], logvar_1.reshape(N, *P, -1)[..., -1]
            mean_2, logvar_2 = self.model(inputs_2_batch.reshape(N, -1, M))
            reward_mean_2, logvar_2 = mean_2.reshape(N, *P, -1)[..., -1], logvar_2.reshape(N, *P, -1)[..., -1]
            logits = reward_mean_1.sum(dim=2) - reward_mean_2.sum(dim=2)
            std = torch.sqrt(torch.cat([torch.exp(logvar_1), torch.exp(logvar_2)], dim=2).sum(dim=2))
            probs = gaussian_cdf(logits / std)
            # Average over batch and dim, sum over ensembles.
            xent_loss_inv = (labels_batch[..., 0] * torch.log(probs + 1e-8) + labels_batch[..., 1] * torch.log(1 - probs + 1e-8)).mean(dim=1)
            var_loss = logvar_1.mean(dim=(1, 2)) + logvar_2.mean(dim=(1, 2))
            loss = xent_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar[-1] - logvar_loss_coef * self.model.min_logvar[-1]

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def validate(self, dynamics_inputs: np.ndarray, dynamics_targets: np.ndarray, rlhf_inputs_1: np.ndarray, rlhf_inputs_2: np.ndarray, rlhf_labels: np.ndarray) -> List[float]:
        self.model.eval()
        dynamics_targets = torch.as_tensor(dynamics_targets).to(self.model.device)
        dynamics_mean, _ = self.model(dynamics_inputs)
        dynamics_loss = ((dynamics_mean[..., :-1] - dynamics_targets) ** 2).mean(dim=(1, 2))

        P, M = rlhf_inputs_1.shape[:-1], rlhf_inputs_1.shape[-1]
        rlhf_mean_1, logvar_1 = self.model(rlhf_inputs_1.reshape(-1, M))
        N = rlhf_mean_1.shape[0]
        rlhf_labels = torch.as_tensor(rlhf_labels).to(self.model.device).unsqueeze(0).repeat(N, *[1] * len(rlhf_labels.shape))
        reward_mean_1, logvar_1 = rlhf_mean_1.reshape(N, *P, -1)[..., -1], logvar_1.reshape(*P, -1)[..., -1]
        rlhf_mean_2, logvar_2 = self.model(rlhf_inputs_2.reshape(-1, M))
        reward_mean_2, logvar_2 = rlhf_mean_2.reshape(N, *P, -1)[..., -1], logvar_2.reshape(*P, -1)[..., -1]
        logits = reward_mean_1.sum(dim=2) - reward_mean_2.sum(dim=2)
        probs = gaussian_cdf(logits)
        rlhf_loss = (rlhf_labels[..., 0] * torch.log(probs + 1e-8) + rlhf_labels[..., 1] * torch.log(1 - probs + 1e-8)).mean(dim=1)
        accuracies = ((logits > 0).float() == rlhf_labels[..., 0]).float().cpu().numpy()
        rewards_win = torch.where((rlhf_labels[..., 0] == 1).unsqueeze(-1), reward_mean_1, reward_mean_2)
        rewards_lose = torch.where((rlhf_labels[..., 1] == 1).unsqueeze(-1), reward_mean_1, reward_mean_2)

        val_loss = list((dynamics_loss + rlhf_loss).cpu().numpy())
        info = {
            'accuracies': accuracies,
            'rewards': torch.cat([reward_mean_1, reward_mean_2], dim=2).cpu().numpy(),
            'rewards_win': rewards_win,
            'rewards_lose': rewards_lose,
        }
        return val_loss, info