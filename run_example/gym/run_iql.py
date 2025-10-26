import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinepbrl.nets import MLP
from offlinepbrl.modules import ActorProb, Critic, DiagGaussian
from offlinepbrl.utils.load_dataset import qlearning_dataset
from offlinepbrl.buffer import ReplayBuffer
from offlinepbrl.utils.logger import Logger, make_log_dirs
from offlinepbrl.policy_trainer import MFPolicyTrainer
from offlinepbrl.policy import IQLPolicy

"""
Task-specific optimized hyperparameters based on CORL benchmark.
These settings match the best-performing configurations from:
https://github.com/tinkoff-ai/CORL

Key insights:
- hopper-medium-expert: needs higher beta (6.0) and lower expectile (0.5)
- hopper-medium-replay: needs slower target updates (tau=0.001)
- All tasks benefit from state normalization
"""

# Optimized hyperparameters for specific tasks (based on CORL)
TASK_CONFIGS = {
    "hopper-medium-expert-v2": {
        "expectile": 0.5,    # More conservative value estimation
        "temperature": 6.0,  # Higher beta for expert data
        "tau": 0.005,
    },
    "hopper-medium-replay-v2": {
        "expectile": 0.7,
        "temperature": 3.0,
        "tau": 0.001,        # Slower target updates for stability
    },
    "hopper-medium-v2": {
        "expectile": 0.7,
        "temperature": 3.0,
        "tau": 0.005,
    },
    # Default settings work well for halfcheetah and walker2d
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="gym")
    parser.add_argument("--algo_name", type=str, default="iql_v2")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_q_lr", type=float, default=3e-4)
    parser.add_argument("--critic_v_lr", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=None, help="Target network update rate. If None, use task-specific default.")
    parser.add_argument("--expectile", type=float, default=None, help="IQL expectile parameter. If None, use task-specific default.")
    parser.add_argument("--temperature", type=float, default=None, help="IQL temperature (beta). If None, use task-specific default.")
    parser.add_argument("--const_reward", type=float, default=None, help="Set all rewards to this constant value.")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    
    # Apply task-specific hyperparameters if not explicitly set
    if args.task in TASK_CONFIGS:
        task_config = TASK_CONFIGS[args.task]
        if args.expectile is None:
            args.expectile = task_config["expectile"]
            print(f"Using task-specific expectile: {args.expectile}")
        if args.temperature is None:
            args.temperature = task_config["temperature"]
            print(f"Using task-specific temperature: {args.temperature}")
        if args.tau is None:
            args.tau = task_config["tau"]
            print(f"Using task-specific tau: {args.tau}")
    else:
        # Use default values for tasks without specific configs
        if args.expectile is None:
            args.expectile = 0.7
        if args.temperature is None:
            args.temperature = 3.0
        if args.tau is None:
            args.tau = 0.005
    
    return args


def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                            dataset["next_observations"][i]
                            ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                        terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    
    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    if args.const_reward is not None:
        dataset["rewards"] = np.full_like(dataset["rewards"], args.const_reward)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    if ("halfcheetah" in args.task or "walker2d" in args.task or "hopper" in args.task):
        dataset = normalize_rewards(dataset)
    
    # Normalize states (CRITICAL for stability!)
    state_mean = dataset["observations"].mean(0)
    state_std = dataset["observations"].std(0) + 1e-3
    dataset["observations"] = (dataset["observations"] - state_mean) / state_std
    dataset["next_observations"] = (dataset["next_observations"] - state_mean) / state_std
    
    # Wrap env to normalize observations during evaluation
    def normalize_state(state):
        return (state - state_mean) / state_std
    env = gym.wrappers.TransformObservation(env, normalize_state)
    
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v = Critic(critic_v_backbone, args.device)
    
    for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.critic_q_lr)
    critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.critic_q_lr)
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)

    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None
    
    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim,
        critic_q1_optim,
        critic_q2_optim,
        critic_v_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # log
    log_dirs = make_log_dirs(args.domain, args.algo_name, args.task, args.seed, vars(args), record_params=["const_reward"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
        eval_freq=args.eval_freq
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()