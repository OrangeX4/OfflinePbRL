# IQL v3 Changes - 进一步修复

## v2 版本的问题分析

从训练日志看到：
```
| misc/adv_mean      | 0.00016  |  # 优势值极小！
| misc/exp_a_mean    | 8.05     |  # exp(0.00016 * 6.0) ≈ 1.001，几乎没有权重
| eval/...reward     | 43.1     |  # 性能远低于目标 107
```

### 🔴 关键问题

1. **Advantage 值太小** (~0.0001)
   - Q ≈ V，导致策略改进信号很弱
   - `exp(adv * beta)` ≈ 1，AWR 权重几乎uniform

2. **Reward Normalization 顺序错误**
   - v2 在状态归一化**之前**计算 trajectory boundaries
   - 使用了**未归一化的状态**来判断 `||s' - next_s|| > 1e-6`
   - 然后又归一化了状态，导致不一致

3. **Actor 更新的 Advantage 计算问题**
   - V network 在 V-update 后已经改变
   - Actor update 时重新计算 V，和 V-update 时的不一致

## v3 版本的修复

### ✅ Fix 1: 调整数据处理顺序

**新顺序**（v3）:
```python
# 1. 先归一化状态
state_mean, state_std = ...
dataset["observations"] = normalize(...)
dataset["next_observations"] = normalize(...)

# 2. 再处理 reward（此时状态已经归一化，trajectory判断更准确）
if locomotion_task:
    dataset = normalize_rewards(dataset)
```

**为什么重要**：
- Reward normalization 需要判断 trajectory 边界
- 边界判断依赖 `||s' - next_s||` 的距离
- 必须使用归一化后的状态，否则距离阈值 `1e-6` 不合适

### ✅ Fix 2: 改进 Reward Normalization

```python
def normalize_rewards(dataset, env_name, max_episode_steps=1000):
    # 使用已归一化的状态来判断边界
    terminals_float = identify_boundaries(dataset)  
    
    # 计算每个 trajectory 的 return
    returns = [compute_return(traj) for traj in trajs]
    min_ret, max_ret = min(returns), max(returns)
    
    # Scale: (r - 0) / (max - min) * 1000
    dataset["rewards"] = dataset["rewards"] / (max_ret - min_ret) * max_episode_steps
    
    # 添加诊断信息
    print(f"Reward range: [{min_ret:.2f}, {max_ret:.2f}] -> "
          f"[{dataset['rewards'].min():.2f}, {dataset['rewards'].max():.2f}]")
```

### ✅ Fix 3: 修正 Actor Update 中的 Advantage 计算

**问题**（v2）:
```python
# V network 已经在 V-update 中更新
critic_v_loss.backward()
critic_v_optim.step()

# ... Q update ...

# Actor update: 重新计算 V（但 V 已经变了！）
v = self.critic_v(obss)  # ← 这个 V 和 V-update 时不同
adv = q - v              # ← Advantage 不准确
```

**修复**（v3）:
```python
# Actor update: 重新计算完整的 advantage
with torch.no_grad():
    # 使用刚更新的 target Q
    target_q = torch.min(
        self.critic_q1_old(obss, actions),
        self.critic_q2_old(obss, actions)
    )
    # 使用当前 V（已经更新过）
    v = self.critic_v(obss)
    # 重新计算 advantage
    adv_actor = target_q - v  # 这是当前最准确的 advantage
    exp_adv = torch.exp(adv_actor * self._temperature)
    exp_adv = torch.clamp(exp_adv, max=100.0)
```

### ✅ Fix 4: 改进 Log Prob 处理

```python
# 明确处理多维 action 的 log_prob
log_probs = dist.log_prob(actions)
if log_probs.dim() > 1:
    log_probs = log_probs.sum(dim=-1, keepdim=True)

actor_loss = -(exp_adv * log_probs).mean()
```

### ✅ Fix 5: 添加更多诊断指标

```python
return {
    "misc/adv_mean": adv.mean().item(),           # V-update 时的 advantage
    "misc/adv_actor_mean": adv_actor.mean().item(), # Actor-update 时的 advantage
    "misc/exp_adv_mean": exp_adv.mean().item(),   # 实际的 AWR 权重
}
```

## 预期改进

### v2 的问题：
- `adv_mean` ≈ 0.0001（太小）
- `exp_a_mean` ≈ 8（但实际权重 ≈ 1）
- 性能 ~40-45

### v3 预期：
- `adv_mean` 应该更大（0.01-0.1 量级）
- `adv_actor_mean` 和 `adv_mean` 应该接近
- `exp_adv_mean` 应该有明显的分化（不是全都接近1）
- 性能应该达到 ~100-110

## 如何验证

运行 v3 后，重点关注：

1. **启动时的 Reward normalization 输出**:
   ```
   Reward normalization: min_return=XXX, max_return=YYY, reward_range=[A, B]
   ```
   - `max_return - min_return` 应该合理（不是超大或超小的数）
   - `reward_range` 应该在 [-几百, +几百] 的范围

2. **训练早期**（前1000步）:
   - `misc/adv_mean` 应该从小逐渐增大
   - `misc/adv_actor_mean` 应该和 `adv_mean` 接近
   - `misc/exp_adv_mean` 应该 > 1（说明有权重分化）

3. **训练中期**（5000-10000步）:
   - `eval/normalized_episode_reward` 应该 > 60
   - `misc/adv_mean` 应该稳定在一个正值

4. **训练后期**（100k+步）:
   - 性能应该稳定在 100 左右
   - 不应该崩溃（像 v1 那样）

## 使用方法

```bash
cd /home/fsj/workspace/OfflinePbRL/run_example/gym

# 自动使用优化的超参数
python run_iql.py --task hopper-medium-expert-v2 --seed 0

# 查看日志，确认：
# 1. 使用了 task-specific 参数（expectile=0.5, temperature=6.0）
# 2. Reward normalization 的输出合理
# 3. adv_actor_mean 有合理的值
```

## 技术细节：为什么这些修改很重要

### 1. 状态归一化必须在前
**原因**: Trajectory boundary 判断使用 `||s' - next_s|| > 1e-6`
- 如果状态未归一化，某些维度的值可能是 1e3 量级，阈值 1e-6 太小
- 如果状态已归一化，所有维度的值在 [-3, 3] 范围，阈值 1e-6 合适

### 2. Advantage 必须一致计算
**原因**: IQL 的 actor update 依赖准确的 advantage
- V network 更新后，V(s) 改变了
- 必须用更新后的 V 重新计算 advantage
- 否则 `exp(adv * beta)` 的权重不准确

### 3. Beta (temperature) 的作用
**物理意义**: `exp(adv * beta)` 是 AWR 权重
- beta=0: uniform 权重（纯行为克隆）
- beta=∞: 只学习最好的 action（最大化 Q）
- beta=6: 对 hopper-expert 的最优值

**为什么需要高 beta (6.0)**:
- Expert data: adv 值小（因为所有 action 都好）
- 需要高 beta 来放大微小的差异
- 否则所有 action 的权重都接近 1，学不到东西

### 4. Expectile 的作用
**物理意义**: Expectile regression 的分位点
- expectile=0.5: mean (MSE)
- expectile=0.7: 更关注 Q > V 的情况（乐观）
- expectile=0.5: 更保守（用于 expert data）

**为什么 expert 需要低 expectile**:
- Expert data: Q 值已经很高
- expectile=0.7 会过高估计 V
- expectile=0.5 更保守，防止 V 超过 Q
