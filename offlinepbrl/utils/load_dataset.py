import os
import pickle
from typing import Optional
import numpy as np
import torch
import collections


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }

def load_queries_with_indices(
        dataset, num_query, len_query, saved_indices, saved_labels=None,
        scripted_teacher=False, relabel_human_labels=False, comparison_equivalence_threshold=0, n_evaluation_categories=5, 
        modality="state", partition_idx=None, feedback_type="comparative"):    
    if modality == "state":
        observation_dim = (dataset["observations"].shape[-1], )
    elif modality == "pixel":
        observation_dim = dataset["observations"].shape[-3:]
    else:
        raise ValueError("Modality error")

    action_dim = dataset["actions"].shape[-1]
    
    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        # do not query all label
        if partition_idx is None:
            query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))
        else:
            # If dataset is large, you should load the dataset in slices.
            query_range = np.arange(partition_idx * num_query, (partition_idx + 1) * num_query)

    total_reward_seq = np.zeros((2, num_query, len_query))
    total_obs_seq = np.zeros((2, num_query, len_query) + observation_dim)
    total_act_seq = np.zeros((2, num_query, len_query, action_dim))
    total_timestep = np.zeros((2, num_query, len_query), dtype=np.int32)

    for query_count, i in enumerate(query_range):
        for j in range(len(saved_indices)):
            start_idx = int(saved_indices[j][i])
            end_idx = start_idx + len_query
            total_reward_seq[j][query_count] = dataset['rewards'][start_idx:end_idx]
            total_obs_seq[j][query_count] = dataset['observations'][start_idx:end_idx]
            total_act_seq[j][query_count] = dataset['actions'][start_idx:end_idx]
            total_timestep[j][query_count] = np.arange(1, len_query + 1)

    batch = {}
    batch['rewards'] = total_reward_seq[0].copy()
    batch['observations'] = total_obs_seq[0].copy()
    batch['actions'] = total_act_seq[0].copy()
    batch['timestep'] = total_timestep[0].copy()
    batch['start_indices'] = saved_indices[0]
    if feedback_type in ['comparative', 'attribute']:
        batch['rewards_2'] = total_reward_seq[1].copy()
        batch['observations_2'] = total_obs_seq[1].copy()
        batch['actions_2'] = total_act_seq[1].copy()
        batch['timestep_2'] = total_timestep[1].copy()
        batch['start_indices_2'] = saved_indices[1]

    if scripted_teacher:
        # scripted labels
        if relabel_human_labels:
            # replace human labels with scripted ones
            if feedback_type in ['comparative']:
                r_t_1 = batch['rewards']
                r_t_2 = batch['rewards_2']
                # reset trajectory rewards
                sum_r_t_1 = np.sum(r_t_1, axis=1)
                sum_r_t_2 = np.sum(r_t_2, axis=1)
                binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
                rational_labels = np.zeros((len(binary_label), 2))
                rational_labels[np.arange(binary_label.size), binary_label] = 1.0
                if comparison_equivalence_threshold > 0.0:
                    margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= comparison_equivalence_threshold).reshape(-1)
                    rational_labels[margin_index] = 0.5
                batch['labels'] = rational_labels
            elif feedback_type in ['evaluative']:
                r_t = batch['rewards']
                # reset trajectory rewards
                sum_r_t = np.sum(r_t, axis=1)
                min_sum_r_t, max_sum_r_t = sum_r_t.min(), sum_r_t.max()
                sum_r_t = (sum_r_t - min_sum_r_t) / (max_sum_r_t - min_sum_r_t) # normalize summed rewards
                evaluation_upper_bounds = np.array([category / (n_evaluation_categories) for category in range(1, n_evaluation_categories + 1)])
                categories = np.clip(np.sum(sum_r_t[:, np.newaxis] >= evaluation_upper_bounds, axis=1), 0, n_evaluation_categories - 1) # category is highest index that is smaller than upper bound
                rational_labels = np.zeros((len(sum_r_t), n_evaluation_categories))
                for i in range(len(rational_labels)):
                    rational_labels[i][categories[i]] = 1
                batch['labels'] = rational_labels
            else:
                raise NotImplementedError('Scripted labels are not supported for "' + feedback_type + '" feedback.')
        else:
            # use already generated fake labels
            batch['labels'] = saved_labels
    else:
        # human labels   
        if feedback_type in ['comparative']:    
            label_shape = (len(saved_labels), 2)
            human_labels = np.zeros(label_shape)
            human_labels[np.array(saved_labels) == 0, 0] = 1.
            human_labels[np.array(saved_labels) == 1, 1] = 1.
            human_labels[np.array(saved_labels) == -1] = 0.5
            human_labels = human_labels[query_range]
            batch['labels'] = human_labels
        elif feedback_type in ['attribute']:
            human_labels = np.array(saved_labels)
            human_labels[np.array(saved_labels) == -1] = 0.5
            human_labels = human_labels[query_range]
            batch['labels'] = human_labels
        elif feedback_type in ['keypoint']:
            human_labels = []
            for i in range(num_query):
                keypoints = []
                for keypoint in saved_labels[saved_indices[0][i]]:
                    keypoints.append(dataset['observations'][keypoint])
                human_labels.append(keypoints)
            batch['labels'] = np.array(human_labels, dtype=object)
        else:
            batch['labels'] = saved_labels

    return batch

def load_rlhf_dataset(
        env,
        dataset=None,
        feedback_type=None, # 'comparative', 'attribute', 'evaluative', 'keypoint', None
        modality: str = 'state', # 'state' or 'pixel'
        num_query: int = 2000,
        len_query: int = 200,
        fake_label: bool = False, # use scripted teacher to label queries instead of human feedback
        relabel_human_labels: bool = False, # if true load human label data and adjust labels with scripted teacher, if false load generated fake label data
        n_evaluation_categories: int = 5,
        comparison_equivalence_threshold: int = 0,
        fake_label_data_dir: str = "../data/generated_fake_labels/",
        human_label_data_dir: str = "../data/crowdsource_human_labels/",
        n_dataset_partition: Optional[int] = None, # 5 for atari dataset
        **kwargs,
    ):
    if dataset is None:
        dataset = qlearning_dataset(env, **kwargs)

    env_name = env.spec.id
    domain2envs = {
        'atari': ['boxing', 'breakout', 'enduro', 'frostbite', 'montezuma_revenge', 'qbert', 'seaquest', 'space_invaders'],
        'mujoco': ['hopper', 'walker2d', 'halfcheetah', 'ant', 'humanoid'],
        'adroit': ['door', 'pen', 'hammer'],
        'antmaze': ['antmaze'],
        'smarts': ['cruise', 'cutin', 'left_c'],
        'd4rl': ['kitchen'],
    }
    domain = ''
    for dom, env_names in domain2envs.items():
        for name in env_names:
            if name in env_name:
                domain = dom
                break
    assert domain != '', f"Domain not found for {env_name}."

    if feedback_type not in ['comparative', 'attribute', 'evaluative', 'keypoint', None]:
        raise NotImplementedError('Learning from the "' + feedback_type + '" feedback type is not supported yet.')

    if not fake_label or relabel_human_labels:
        data_dir = os.path.join(os.path.dirname(__file__), human_label_data_dir)
        suffix = "_human_labels"
    else:
        data_dir = os.path.join(os.path.dirname(__file__), fake_label_data_dir)
        suffix = "_fake_labels"

    if feedback_type:
        data_dir = os.path.join(data_dir, f"{env_name}" + suffix, feedback_type)
    else:
        data_dir = os.path.join(data_dir, f"{env_name}" + suffix)

    if feedback_type is None:
        # use comparative feedback by default
        feedback_type = "comparative"

    print(f"Load saved indices from {data_dir}.")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Label not found for {env_name} in {data_dir}.")
    
    suffix = f"domain_{domain}_env_{env_name}_num_{num_query}_len_{len_query}"
    matched_file = []
    for file_name in os.listdir(data_dir):
        print(suffix)
        print(file_name)
        if suffix in file_name:
            matched_file.append(file_name)

    if len(matched_file) == 0:
        raise ValueError(f"No matching labels found in {data_dir} for {suffix}.")
    
    # unpickle transformed labels
    unpickled_data = {}
    for file in matched_file:
        file_name = os.path.splitext(os.path.basename(file))[0]
        data_type = file_name.split('_domain_')[0]
        identifier = file_name.split('_')[-1]

        if identifier not in unpickled_data:
            unpickled_data[identifier] = {}
        with open(os.path.join(data_dir, file), "rb") as fp:  # Unpickling
            unpickled_data[identifier][data_type] = pickle.load(fp)
        
        if 'query_length' not in unpickled_data[identifier]:
            unpickled_data[identifier]['query_length'] = int(file_name.split('_len_')[1].split('_')[0])

    # verify that all datasets have the same number of queries
    query_length = next(iter(unpickled_data.values()))['query_length']
    for identifier in unpickled_data:
        assert unpickled_data[identifier]['query_length'] == query_length
        unpickled_data[identifier].pop('query_length')

    # concat data if multiple datasets are given
    concatenated_unpickled_data = {}
    for identifier in unpickled_data:
        for data_type in unpickled_data[identifier]:
            if data_type not in concatenated_unpickled_data:
                if isinstance(unpickled_data[identifier][data_type], dict):
                    concatenated_unpickled_data[data_type] = {}
                else:
                    initial_shape = (0,)
                    if len(unpickled_data[identifier][data_type].shape) > 1:
                        initial_shape += unpickled_data[identifier][data_type].shape[1:]
                    concatenated_unpickled_data[data_type] = np.empty(initial_shape)
            if isinstance(unpickled_data[identifier][data_type], dict):
                concatenated_unpickled_data[data_type] = {
                    **concatenated_unpickled_data[data_type],
                    **unpickled_data[identifier][data_type]
                }
            else:
                concatenated_unpickled_data[data_type] = np.concatenate((
                    concatenated_unpickled_data[data_type],
                    unpickled_data[identifier][data_type]
                ))

    # verify that the entries of all data types have the same length
    assert all(len(value) == len(next(iter(concatenated_unpickled_data.values()))) for value in concatenated_unpickled_data.values())

    # add query length and query number to the output
    concatenated_unpickled_data['num_query'] = len(next(iter(concatenated_unpickled_data.values())))
    concatenated_unpickled_data['len_query'] = query_length

    label_data = ()
    for data_type in [
        'indices', 'indices_1', 'indices_2', 
        'human_label', 'fake_label',
        'num_query', 'len_query'
    ]:
        if data_type in concatenated_unpickled_data:
            label_data = label_data + (concatenated_unpickled_data[data_type],)
    label_data = label_data[0] if len(label_data) == 1 else label_data

    if feedback_type in ['comparative', 'attribute']:
        human_indices_1, human_indices_2, human_labels, num_query, len_query = label_data
    elif feedback_type in ['evaluative']:
        human_indices, human_labels, num_query, len_query = label_data
    elif feedback_type in ['keypoint', 'visual']:
        human_labels, num_query, len_query = label_data
    else:
        raise ValueError("Invalid feedback type:", feedback_type)

    if feedback_type in ["comparative", "attribute"]:
        saved_indices = [human_indices_1, human_indices_2]
    elif feedback_type in ["evaluative", "visual"]:
        saved_indices = [human_indices]
    elif feedback_type in ['keypoint']:
        saved_indices = [np.array(list(human_labels.keys()))]

    if n_dataset_partition is None:
        rlhf_dataset = load_queries_with_indices(
            dataset, num_query, len_query, saved_indices, saved_labels=human_labels, scripted_teacher=fake_label, 
            relabel_human_labels=relabel_human_labels, comparison_equivalence_threshold=comparison_equivalence_threshold, 
            n_evaluation_categories=n_evaluation_categories, modality=modality, feedback_type=feedback_type)
    else:
        assert type(n_dataset_partition) is int, "n_dataset_partition should be int"
        rlhf_dataset = [load_queries_with_indices(
            dataset, num_query // n_dataset_partition, len_query, saved_indices=saved_indices, saved_labels=human_labels, scripted_teacher=fake_label, 
            relabel_human_labels=relabel_human_labels, comparison_equivalence_threshold=comparison_equivalence_threshold,
            n_evaluation_categories=n_evaluation_categories, modality=modality, partition_idx=p_idx, feedback_type=feedback_type)
            for p_idx in range(n_dataset_partition)]

    return rlhf_dataset


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, max_ep_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = torch.device(device)
        self.input_mean = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).mean(0)
        self.input_std = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        
        indices = []
        for traj_ind, traj in enumerate(self.trajs):
            end = len(traj["rewards"])
            for i in range(end):
                indices.append((traj_ind, i, i+self.max_len))

        self.indices = np.array(indices)
        

        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_ind, start_ind, end_ind = self.indices[idx]
        traj = self.trajs[traj_ind].copy()
        obss = traj['observations'][start_ind:end_ind]
        actions = traj['actions'][start_ind:end_ind]
        next_obss = traj['next_observations'][start_ind:end_ind]
        rewards = traj['rewards'][start_ind:end_ind].reshape(-1, 1)
        delta_obss = next_obss - obss
    
        # padding
        tlen = obss.shape[0]
        inputs = np.concatenate([obss, actions], axis=1)
        inputs = (inputs - self.input_mean) / self.input_std
        inputs = np.concatenate([inputs, np.zeros((self.max_len - tlen, self.obs_dim+self.action_dim))], axis=0)
        targets = np.concatenate([delta_obss, rewards], axis=1)
        targets = np.concatenate([targets, np.zeros((self.max_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.max_len - tlen)], axis=0)

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32, device=self.device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=self.device)

        return inputs, targets, masks