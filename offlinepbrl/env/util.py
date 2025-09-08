import gym
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit
from offlinepbrl.env.wrappers import NormalizedBoxEnv


def make_metaworld_env(env_name, seed):
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    # print("partially observe", env._partially_observable) Ture
    # print("env._freeze_rand_vec", env._freeze_rand_vec) True
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)