import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinepbrl.nets import MLP
from offlinepbrl.modules import ActorProb, Critic, DiagGaussian
from offlinepbrl.modules.reward_module import RewardModel, EnsembleRewardModel
from offlinepbrl.utils.load_metaworld_dataset import load_metaworld_mr_dataset, collect_feedback
from offlinepbrl.env.util import make_metaworld_env
from offlinepbrl.buffer import ReplayBuffer, PrefBuffer
from offlinepbrl.utils.logger import Logger, make_log_dirs
from offlinepbrl.policy_trainer import MFPolicyTrainer
from offlinepbrl.policy import BTWrapper, IQLPolicy

"""
suggested hypers
expectile=0.7, temperature=3.0 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="bt_iql_metaworld")
    parser.add_argument("--task", type=str, default="metaworld_box-close-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_q_lr", type=float, default=3e-4)
    parser.add_argument("--critic_v_lr", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    
    # BT specific parameters
    parser.add_argument("--reward_model_lr", type=float, default=3e-4)
    parser.add_argument("--reward_activation", type=str, default="tanh", choices=["identity", "sigmoid", "tanh", "relu", "leaky_relu"])
    parser.add_argument("--reward_reg", type=float, default=0.0)
    parser.add_argument("--rm_stop_epoch", type=int, default=None)
    parser.add_argument("--policy_start_epoch", type=int, default=None)
    parser.add_argument("--ensemble_num", type=int, default=3)
    
    # data collection
    parser.add_argument("--data_quality", type=float, default=8.0)
    parser.add_argument("--feedback_num", type=int, default=1000)
    parser.add_argument("--segment_size", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--feedback_type", type=str, default="RLT")
    
    parser.add_argument("--epoch", type=int, default=1200)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pref_batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = make_metaworld_env(args.task, args.seed)
    dataset = load_metaworld_mr_dataset(args.task, args.data_quality)
    
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
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
    
    # create preference buffer
    traj_total = len(dataset["observations"]) // 500
    feedback = collect_feedback(
        dataset, 
        traj_total, 
        args.feedback_num, 
        args.segment_size, 
        args.feedback_type, 
        args.threshold, 
        args.noise
    )
    
    # Convert feedback format to what the framework expects
    idx_st_1 = []
    idx_st_2 = []
    labels = []
    # construct the preference pairs from multiple_ranked_list
    for single_ranked_list in feedback:
        for i in range(len(single_ranked_list)):
            for j in range(i + 1, len(single_ranked_list)):
                group_i = single_ranked_list[i] 
                group_j = single_ranked_list[j]
                for item_i in group_i:
                    for item_j in group_j:
                        idx_st_1.append(item_i[0])
                        idx_st_2.append(item_j[0])
                        labels.append([0, 1])  # group_j (later in list) is preferred over group_i
    labels = np.array(labels)
    
    # Convert to segment indices
    idx_1 = [[j for j in range(i, i + args.segment_size)] for i in idx_st_1]
    idx_2 = [[j for j in range(i, i + args.segment_size)] for i in idx_st_2]
    
    obs_1_s = dataset["observations"][np.array(idx_1)]
    action_1_s = dataset["actions"][np.array(idx_1)]
    obs_2_s = dataset["observations"][np.array(idx_2)]
    action_2_s = dataset["actions"][np.array(idx_2)]
    
    # Convert to format expected by PrefBuffer
    # Data should be in format [num_pairs, sequence_length, feature_dim]
    num_pairs = len(obs_1_s)
    
    # Create timestep arrays (1 to segment_size for each trajectory)
    timesteps_1 = np.tile(np.arange(1, args.segment_size + 1), (num_pairs, 1))
    timesteps_2 = np.tile(np.arange(1, args.segment_size + 1), (num_pairs, 1))
    
    # Create start indices (marks beginning of each trajectory)
    start_indices_1 = np.arange(num_pairs) * args.segment_size
    start_indices_2 = np.arange(num_pairs) * args.segment_size
    
    # Create dummy rewards (will be replaced by reward model)
    rewards_1 = np.zeros((num_pairs, args.segment_size))
    rewards_2 = np.zeros((num_pairs, args.segment_size))
    
    rlhf_dataset = {
        "observations": obs_1_s,
        "actions": action_1_s,
        "rewards": rewards_1,
        "timestep": timesteps_1,
        "start_indices": start_indices_1,
        "observations_2": obs_2_s,
        "actions_2": action_2_s,
        "rewards_2": rewards_2,
        "timestep_2": timesteps_2,
        "start_indices_2": start_indices_2,
        "labels": labels
    }

    pref_buffer = PrefBuffer(
        buffer_size=len(rlhf_dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        max_traj_len=rlhf_dataset["observations"].shape[1],
        device=args.device
    )
    pref_buffer.load_dataset(rlhf_dataset)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=False,
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v = Critic(critic_v_backbone, args.device)
    
    reward_model_backbones = [
        MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims) 
        for _ in range(args.ensemble_num)
    ]
    reward_models = [
        RewardModel(backbone, activation=args.reward_activation, device=args.device)
        for backbone in reward_model_backbones
    ]
    reward_model = EnsembleRewardModel(reward_models, device=args.device)
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=args.critic_q_lr)
    critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=args.critic_q_lr)
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)
    reward_model_optim = torch.optim.Adam(reward_model.parameters(), lr=args.reward_model_lr)

    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None
    
    # create IQL policy
    base_policy = IQLPolicy(
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
    
    # Wrap with BT
    policy = BTWrapper(
        base_policy=base_policy,
        reward_model=reward_model,
        reward_model_optim=reward_model_optim,
        reward_reg=args.reward_reg,
        rm_stop_epoch=args.rm_stop_epoch,
        policy_start_epoch=args.policy_start_epoch
    )

    # log
    log_dirs = make_log_dirs(args.algo_name, args.task, args.seed, vars(args))
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
        pref_buffer=pref_buffer,
        pref_batch_size=args.pref_batch_size,
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()
