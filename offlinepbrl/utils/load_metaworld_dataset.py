import pickle
from typing import Optional
import numpy as np
import gym, random, torch, os, uuid

def get_indices(traj_total, segment_size):
    traj_idx = np.random.choice(traj_total, replace=False)
    idx_st = 500 * traj_idx + np.random.randint(0, 500 - segment_size)
    idx = [[j for j in range(idx_st, idx_st + segment_size)]]
    return idx


def consist_test_dataset(
    dataset, test_feedback_num, traj_total, segment_size, threshold
):
    test_traj_idx = np.random.choice(traj_total, 2 * test_feedback_num, replace=True)
    test_idx = [
        500 * i + np.random.randint(0, 500 - segment_size) for i in test_traj_idx
    ]
    test_idx_st_1 = test_idx[:test_feedback_num]
    test_idx_st_2 = test_idx[test_feedback_num:]
    test_idx_1 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_1]
    test_idx_2 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_2]
    test_labels = obtain_labels(
        dataset,
        test_idx_1,
        test_idx_2,
        segment_size=segment_size,
        threshold=threshold,
        noise=0.0,
    )
    test_binary_labels = obtain_labels(
        dataset,
        test_idx_1,
        test_idx_2,
        segment_size=segment_size,
        threshold=0,
        noise=0.0,
    )
    test_obs_act_1 = np.concatenate(
        (dataset["observations"][test_idx_1], dataset["actions"][test_idx_1]),
        axis=-1,
    )
    test_obs_act_2 = np.concatenate(
        (dataset["observations"][test_idx_2], dataset["actions"][test_idx_2]),
        axis=-1,
    )
    return test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels


def collect_feedback(dataset, traj_total, feedback_num, segment_size=25, feedback_type="RLT", threshold=0.5, noise=0.0, q_budget=1):
    multiple_ranked_list = []
    print("feedback_num", feedback_num)
    if feedback_type == "RLT":
        # If q_budget = 1, we collect independent pairwise feedback
        print("Construct RLT")
        cum_query = 0
        current_query_num = 0
        single_ranked_list = []
        while cum_query < feedback_num:
            if current_query_num >= q_budget:
                multiple_ranked_list.append(single_ranked_list)
                current_query_num = 0
                single_ranked_list = []
            # binary search
            group = []
            if len(single_ranked_list) == 0:
                idx_1 = get_indices(traj_total, segment_size)
                idx_2 = get_indices(traj_total, segment_size)
                idx_st_1 = idx_1[0][0]
                idx_st_2 = idx_2[0][0]
                reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
                reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
                label = obtain_labels(
                    dataset,
                    idx_1,
                    idx_2,
                    segment_size=segment_size,
                    threshold=threshold,
                    noise=noise,
                )
                cum_query += 1
                current_query_num += 1
                if np.all(label[0] == [1, 0]):
                    group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                elif np.all(label[0] == [0.5, 0.5]):
                    group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
                    single_ranked_list.append(group)
                elif np.all(label[0] == [0, 1]):
                    group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
            else:
                idx = get_indices(traj_total, segment_size)
                idx_st = idx[0][0]
                reward = np.round(np.sum(dataset["rewards"][idx], axis=1), 2).item()
                low = 0
                high = len(single_ranked_list)
                pos = 0
                insert = False
                while low < high:
                    mid = (low + high) // 2
                    mid_num = (low + high) // 2
                    mid = (mid + mid_num) // 2
                    len_mid = len(single_ranked_list[mid])
                    # random select one element in the mid group
                    random_idx = np.random.randint(0, len_mid)
                    compare_idx = single_ranked_list[mid][random_idx][0]
                    compare_seg_idx = [
                        [
                            j
                            for j in range(
                                compare_idx, compare_idx + segment_size
                            )
                        ]
                    ]
                    label = obtain_labels(
                        dataset,
                        idx,
                        compare_seg_idx,
                        segment_size=segment_size,
                        threshold=threshold,
                        noise=noise,
                    )
                    cum_query += 1
                    current_query_num += 1
                    if np.all(label[0] == [0, 1]):
                        high = mid
                        pos = mid
                    elif np.all(label[0] == [1, 0]):
                        low = mid + 1
                        pos = mid + 1
                    else:
                        if cum_query <= feedback_num:
                            single_ranked_list[mid].append((idx_st, reward))
                        insert = True
                        break
                if insert == False and cum_query <= feedback_num:
                    single_ranked_list.insert(pos, [(idx_st, reward)])
        multiple_ranked_list.append(single_ranked_list)
        return multiple_ranked_list
    elif feedback_type == "SeqRank":
        print("Sequential Pairwise feedback (SeqRank)")
        single_ranked_list = []
        up = 0
        if len(single_ranked_list) == 0:
            idx_1 = get_indices(traj_total, segment_size)
            idx_2 = get_indices(traj_total, segment_size)
            idx_st_1 = idx_1[0][0]
            idx_st_2 = idx_2[0][0]
            reward_1 = np.round(np.sum(dataset["rewards"][idx_1], axis=1), 2).item()
            reward_2 = np.round(np.sum(dataset["rewards"][idx_2], axis=1), 2).item()
            label = obtain_labels(
                dataset,
                idx_1,
                idx_2,
                segment_size=segment_size,
                threshold=threshold,
                noise=noise,
            )
            if np.all(label[0] == [1, 0]):
                group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                single_ranked_list.append(group_2)
                single_ranked_list.append(group_1)
                up = -1
            elif np.all(label[0] == [0.5, 0.5]):
                group = [(idx_st_1, reward_1), (idx_st_2, reward_2)]
                single_ranked_list.append(group)
                up = 0
            elif np.all(label[0] == [0, 1]):
                group_1, group_2 = [(idx_st_1, reward_1)], [(idx_st_2, reward_2)]
                single_ranked_list.append(group_1)
                single_ranked_list.append(group_2)
                up = 1
        last_idx = idx_2
        last_idx_st = last_idx[0][0]
        for i in range(feedback_num - 1):
            idx = get_indices(traj_total, segment_size)
            idx_st = idx[0][0]
            reward_last = np.round(
                np.sum(dataset["rewards"][last_idx], axis=1), 2
            ).item()
            reward = np.round(np.sum(dataset["rewards"][idx], axis=1), 2).item()
            label = obtain_labels(
                dataset,
                last_idx,
                idx,
                segment_size=segment_size,
                threshold=threshold,
                noise=noise,
            )
            if np.all(label[0] == [1, 0]):
                group_1, group_2 = [(last_idx_st, reward_last)], [(idx_st, reward)]
                curr_up = -1
                if up == curr_up or up == 0:
                    # insert front of single_ranked_list
                    single_ranked_list.insert(0, group_2)
                    up = -1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                    up = -1
            elif np.all(label[0] == [0.5, 0.5]):
                if up == -1:
                    single_ranked_list[0].append((idx_st, reward))
                else:
                    single_ranked_list[-1].append((idx_st, reward))
            elif np.all(label[0] == [0, 1]):
                group_1, group_2 = [(last_idx_st, reward_last)], [(idx_st, reward)]
                curr_up = 1
                if up == curr_up or up == 0:
                    single_ranked_list.append(group_2)
                    up = 1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
                    up = 1
            last_idx = idx
            last_idx_st = idx[0][0]
        multiple_ranked_list.append(single_ranked_list)
        return multiple_ranked_list


def collect_human_feedback(dataset, env_name, feedback_type="RLT", segment_size=25, q_budget=1):
    print("Human feedback")
    multiple_ranked_list = []
    if feedback_type == "RLT":
        # indepednent sampling
        if q_budget == 1:
            print("Independent pairwise feedback")
            path = f"./human_feedback/{env_name}/_Independent.txt"
            with open(path, "r") as f:
                for line in f:
                    single_ranked_list = []
                    line = line.split(" ")
                    idx_1 = int(line[0])
                    idx_2 = int(line[1])
                    label = int(line[2])
                    reward_1 = np.round(
                        np.sum(dataset["rewards"][idx_1 : idx_1 + segment_size]),
                        2,
                    ).item()
                    reward_2 = np.round(
                        np.sum(dataset["rewards"][idx_2 : idx_2 + segment_size]),
                        2,
                    ).item()
                    if label == 1:
                        group_1, group_2 = [(idx_1, reward_1)], [(idx_2, reward_2)]
                        single_ranked_list.append(group_2)
                        single_ranked_list.append(group_1)
                    elif label == 2:
                        group = [(idx_1, reward_1), (idx_2, reward_2)]
                        single_ranked_list.append(group)
                    elif label == 3:
                        group_1, group_2 = [(idx_1, reward_1)], [(idx_2, reward_2)]
                        single_ranked_list.append(group_1)
                        single_ranked_list.append(group_2)
                    multiple_ranked_list.append(single_ranked_list)
        # LiRE
        elif q_budget > 1:
            print("Construct RLT (LiRE)")
            for s in range(0, 2):
                path = f"./human_feedback/{env_name}/_RLT_{s}.txt"
                single_ranked_list = []
                with open(path, "r") as f:
                    for line in f:
                        line = line.split("[")[1].split("]")[0].split(", ")
                        group = []
                        for idx in line:
                            idx = int(idx)
                            group.append(
                                (
                                    idx,
                                    np.round(
                                        np.sum(
                                            dataset["rewards"][
                                                idx : idx + segment_size
                                            ]
                                        ),
                                        2,
                                    ),
                                )
                            )
                        single_ranked_list.append(group)
                multiple_ranked_list.append(single_ranked_list)

    elif feedback_type == "SeqRank":
        print("Sequential Pairwise feedback (SeqRank)")
        path = f"./human_feedback/{env_name}/_SeqRank.txt"
        single_ranked_list = []
        idx_st_1 = []
        idx_st_2 = []
        labels = []
        raw_labels = []
        reward_1 = []
        reward_2 = []
        with open(path, "r") as f:
            for line in f:
                line = line.split("\t")
                index_1 = int(line[0])
                index_2 = int(line[1])
                label = int(line[2])
                idx_st_1.append(index_1)
                idx_st_2.append(index_2)
                raw_labels.append(label)
            idx_1 = [[j for j in range(i, i + segment_size)] for i in idx_st_1]
            idx_2 = [[j for j in range(i, i + segment_size)] for i in idx_st_2]
            reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
            reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
            for i in range(len(raw_labels)):
                if raw_labels[i] == 1:
                    labels.append([1, 0])
                elif raw_labels[i] == 2:
                    labels.append([0.5, 0.5])
                elif raw_labels[i] == 3:
                    labels.append([0, 1])
            labels = np.array(labels)
        if np.all(labels[0] == [1, 0]):
            group_1, group_2 = [(idx_st_1[0], reward_1[0])], [
                (idx_st_2[0], reward_2[0])
            ]
            single_ranked_list.append(group_2)
            single_ranked_list.append(group_1)
            up = -1
        elif np.all(labels[0] == [0.5, 0.5]):
            group = [(idx_st_1[0], reward_1[0]), (idx_st_2[0], reward_2[0])]
            single_ranked_list.append(group)
            up = 0
        elif np.all(labels[0] == [0, 1]):
            group_1, group_2 = [(idx_st_1[0], reward_1[0])], [
                (idx_st_2[0], reward_2[0])
            ]
            single_ranked_list.append(group_1)
            single_ranked_list.append(group_2)
            up = 1
        for i in range(1, len(labels)):
            if np.all(labels[i] == [1, 0]):
                group_1, group_2 = [(idx_st_1[i], reward_1[i])], [
                    (idx_st_2[i], reward_2[i])
                ]
                curr_up = -1
                if up == curr_up or up == 0:
                    # insert front of single_ranked_list
                    single_ranked_list.insert(0, group_2)
                    up = -1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_2)
                    single_ranked_list.append(group_1)
                    up = -1
            elif np.all(labels[i] == [0.5, 0.5]):
                if up == -1:
                    single_ranked_list[0].append((idx_st_2[i], reward_2[i]))
                else:
                    single_ranked_list[-1].append((idx_st_2[i], reward_2[i]))
            elif np.all(labels[i] == [0, 1]):
                group_1, group_2 = [(idx_st_1[i], reward_1[i])], [
                    (idx_st_2[i], reward_2[i])
                ]
                curr_up = 1
                if up == curr_up or up == 0:
                    single_ranked_list.append(group_2)
                    up = 1
                else:
                    multiple_ranked_list.append(single_ranked_list)
                    single_ranked_list = []
                    single_ranked_list.append(group_1)
                    single_ranked_list.append(group_2)
                    up = 1
        multiple_ranked_list.append(single_ranked_list)
    return multiple_ranked_list


def obtain_labels(dataset, idx_1, idx_2, segment_size=25, threshold=0.5, noise=0.0):
    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)
    labels = []
    reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
    reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)
    labels = np.where(reward_1 < reward_2, 1, 0)
    labels = np.array([[1, 0] if i == 0 else [0, 1] for i in labels]).astype(float)
    gap = segment_size * threshold

    equal_labels = np.where(
        np.abs(reward_1 - reward_2) <= segment_size * threshold, 1, 0
    )
    labels = np.array(
        [labels[i] if equal_labels[i] == 0 else [0.5, 0.5] for i in range(len(labels))]
    )
    if noise != 0.0:
        p = noise
        for i in range(len(labels)):
            if labels[i][0] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            elif labels[i][1] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 1
                        labels[i][1] = 0
                    else:
                        labels[i][0] = 0.5
                        labels[i][1] = 0.5
            else:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i][0] = 0
                        labels[i][1] = 1
                    else:
                        labels[i][0] = 1
                        labels[i][1] = 0
    return labels


def load_metaworld_mr_dataset(env_name: str, data_quality: float = 1.0, human: bool = False, base_path: Optional[str] = None):
    """
    MetaWorld medium-replay dataset from LiRE (Choi et al., 2024)
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "../data/metaworld/MetaWorld_medium-replay/" + env_name.split("_")[1])
    
    if human == False:
        dataset = dict()
        for seed in range(3):
            path = os.path.join(base_path, f"saved_replay_buffer_1000000_seed{seed}.pkl")
            with open(path, "rb") as f:
                load_dataset = pickle.load(f)

            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][: int(data_quality * 100_000)]
            load_dataset["terminals"] = load_dataset["dones"][: int(data_quality * 100_000)]
            load_dataset.pop("dones", None)

            for key in load_dataset.keys():
                if key not in dataset:
                    dataset[key] = load_dataset[key]
                else:
                    dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)
    elif human == True:
        # Assuming similar for human, but adjust if needed
        path = os.path.join(base_path, "human_trajectory.pkl")
        with open(path, "rb") as f:
            dataset = pickle.load(f)
            dataset["observations"] = np.array(dataset["observations"])
            dataset["actions"] = np.array(dataset["actions"])
            dataset["next_observations"] = np.array(dataset["next_observations"])
            dataset["rewards"] = np.array(dataset["rewards"])
            dataset["terminals"] = np.array(dataset["dones"])

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def load_metaworld_me_dataset(env_name: str, data_quality: float = 1.0, base_path: Optional[str] = None):
    """
    MetaWorld medium-expert dataset following the approaches of IPL (Hejna & Sadigh, 2024) and LiRE (Choi et al., 2024) 
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "../data/metaworld/MetaWorld_medium-expert/" + env_name.split("_")[1])
    
    load_dataset = np.load(os.path.join(base_path, "trajectory.npz"))
    dataset = {key: load_dataset[key] for key in load_dataset.keys()}

    if data_quality * 100_000 >= dataset["rewards"].shape[0]:
        idx = np.arange(dataset["rewards"].shape[0]//500)
    else:
        N = dataset["rewards"].shape[0] // 500  # 600 for metaworld medium-expert
        # take trajectories proportional to the data quality
        n_expert = int(data_quality * 200 / 12)
        idx_expert = np.arange(n_expert)
        n_within_env = int(data_quality * 200 / 12)
        idx_within_env = np.arange(n_within_env) + int(N/12)
        n_random = int(data_quality * 200 / 3)
        idx_random = np.arange(n_random) + int(N/6)
        n_eps_greedy = int(data_quality * 200 / 3)
        idx_eps_greedy = np.arange(n_eps_greedy) + int(N/3)
        n_cross_env = int(data_quality * 200 / 6)
        idx_cross_env = np.arange(n_cross_env) + int(2*N/3)

        idx = np.concatenate((idx_expert, idx_within_env, idx_random, idx_eps_greedy, idx_cross_env), axis=0)
    
    state_dim = dataset["states"].shape[1]
    action_dim = dataset["actions"].shape[1]

    return {
        "observations": dataset["states"].astype(np.float32).reshape(-1,500,state_dim)[idx].reshape(-1,state_dim),
        "actions": dataset["actions"].astype(np.float32).reshape(-1,500,action_dim)[idx].reshape(-1,action_dim),
        "next_observations": dataset["next_states"].astype(np.float32).reshape(-1,500,state_dim)[idx].reshape(-1,state_dim),
        "rewards": dataset["rewards"].astype(np.float32).reshape(N,500,-1)[idx].reshape(-1),
        "terminals": dataset["dones"].astype(bool).reshape(N,500,-1)[idx].reshape(-1),
    }
