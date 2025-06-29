import argparse
import random

import gym
import d4rl

import numpy as np
import torch


from offlinepbrl.nets import MLP
from offlinepbrl.modules import ActorProb, Critic, TanhDiagGaussian
from offlinepbrl.modules.reward_module import GaussianRewardModel
from offlinepbrl.utils.load_dataset import qlearning_dataset, load_rlhf_dataset
from offlinepbrl.buffer import ReplayBuffer, PrefBuffer
from offlinepbrl.utils.logger import Logger, make_log_dirs
from offlinepbrl.policy_trainer import MFPolicyTrainer
from offlinepbrl.policy import CQLPolicy
from offlinepbrl.policy.preference.gtm import GaussianTMWrapper

"""
Gaussian Thurstone-Mosteller with CQL
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="gtm_cql")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    # Gaussian TM specific parameters
    parser.add_argument("--reward-model-lr", type=float, default=3e-4)
    parser.add_argument("--reward-activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "relu", "leaky_relu"])
    parser.add_argument("--reward-reg", type=float, default=0.0)
    parser.add_argument("--reward-ent-reg", type=float, default=0.1)
    parser.add_argument("--entropy-threshold", type=float, default=0.1)
    parser.add_argument("--reg-type", type=str, default="transition", choices=["transition", "trajectory"])
    parser.add_argument("--rm-stop-epoch", type=int, default=200)
    parser.add_argument("--policy-start-epoch", type=int, default=200)
    
    parser.add_argument("--epoch", type=int, default=1200)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pref-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    rlhf_dataset = load_rlhf_dataset(env, dataset)
    
    # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    if 'antmaze' in args.task:
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 4.0
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
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    reward_model_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    
    # Use Gaussian reward model for Thurstone-Mosteller learning
    reward_model = GaussianRewardModel(reward_model_backbone, activation=args.reward_activation, device=args.device)
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    reward_model_optim = torch.optim.Adam(reward_model.parameters(), lr=args.reward_model_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create CQL policy
    base_policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions
    )
    
    # Wrap with Gaussian TM
    policy = GaussianTMWrapper(
        base_policy=base_policy,
        reward_model=reward_model,
        reward_model_optim=reward_model_optim,
        reward_reg=args.reward_reg,
        reward_ent_reg=args.reward_ent_reg,
        entropy_threshold=args.entropy_threshold,
        reg_type=args.reg_type,
        rm_stop_epoch=args.rm_stop_epoch,
        policy_start_epoch=args.policy_start_epoch
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

    # create preference buffer
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
        pref_buffer=pref_buffer,
        pref_batch_size=args.pref_batch_size
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()