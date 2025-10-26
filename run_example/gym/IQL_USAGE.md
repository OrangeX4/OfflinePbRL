# IQL 使用说明

## 快速开始

IQL 现在会**自动为不同任务选择最优超参数**，无需手动配置！

### 基本用法

```bash
# 直接运行，自动使用最优超参数
python run_iql.py --task hopper-medium-expert-v2 --seed 0

# 指定不同的任务
python run_iql.py --task hopper-medium-replay-v2 --seed 0
python run_iql.py --task walker2d-medium-expert-v2 --seed 0
python run_iql.py --task halfcheetah-medium-expert-v2 --seed 0
```

## 自动超参数配置

以下任务会自动应用 CORL 基准测试中的最优超参数：

| 任务 | expectile | temperature (β) | tau | 说明 |
|-----|-----------|----------------|-----|------|
| `hopper-medium-expert-v2` | **0.5** | **6.0** | 0.005 | 更保守的价值估计 + 更强的策略改进 |
| `hopper-medium-replay-v2` | 0.7 | 3.0 | **0.001** | 更慢的目标网络更新提高稳定性 |
| `hopper-medium-v2` | 0.7 | 3.0 | 0.005 | 标准配置 |
| 其他任务 | 0.7 | 3.0 | 0.005 | 默认配置（适用于大部分任务） |

### 运行时提示

脚本会在运行时打印使用的超参数：
```
Using task-specific expectile: 0.5
Using task-specific temperature: 6.0
Using task-specific tau: 0.005
```

## 手动覆盖超参数

如果需要实验不同的超参数，可以手动指定：

```bash
# 手动指定会覆盖自动配置
python run_iql.py \
    --task hopper-medium-expert-v2 \
    --expectile 0.6 \
    --temperature 5.0 \
    --tau 0.01 \
    --seed 0
```

## 重要改进（已修复的问题）

✅ **状态归一化**: 现在自动对状态进行归一化，显著提高训练稳定性  
✅ **正确的更新顺序**: V → Q → 同步目标网络 → Actor  
✅ **任务特定超参数**: 为 hopper 任务优化了超参数  
✅ **改进的日志**: 添加了 `adv_mean`, `exp_a_mean`, `v` 等诊断指标  

## 预期性能

使用自动超参数配置后，应该能达到以下性能（D4RL normalized score）：

- `hopper-medium-expert-v2`: ~100-110 (CORL: 107.42 ± 7.80)
- `hopper-medium-replay-v2`: ~95-100 (CORL: 97.43 ± 6.39)
- `hopper-medium-v2`: ~65-70 (CORL: 67.53 ± 3.78)

## 完整参数列表

```bash
python run_iql.py \
    --task hopper-medium-expert-v2 \  # 任务名称
    --seed 0 \                         # 随机种子
    --expectile 0.5 \                  # IQL expectile (None=自动)
    --temperature 6.0 \                # IQL temperature/beta (None=自动)
    --tau 0.005 \                      # 目标网络更新率 (None=自动)
    --actor_lr 3e-4 \                  # Actor 学习率
    --critic_q_lr 3e-4 \               # Q-critic 学习率
    --critic_v_lr 3e-4 \               # V-critic 学习率
    --gamma 0.99 \                     # 折扣因子
    --batch_size 256 \                 # 批量大小
    --epoch 1000 \                     # 训练轮数
    --step_per_epoch 1000 \            # 每轮步数
    --eval_freq 5 \                    # 评估频率（每N轮）
    --eval_episodes 10 \               # 评估episode数
    --lr_decay True \                  # 是否使用余弦学习率衰减
    --device cuda                      # 设备
```

## 监控训练

关注以下指标判断训练是否正常：

### 正常训练的信号
- ✅ `eval/normalized_episode_reward` 稳定上升
- ✅ `misc/adv_mean` 保持正值且稳定
- ✅ `misc/exp_a_mean` 在 [1, 100] 范围内
- ✅ `misc/v` 和 `misc/q1` 平滑增长

### 异常训练的信号
- ❌ 性能突然崩溃（reward 骤降）
- ❌ `exp_a_mean` 频繁达到 100.0 上限
- ❌ Q 值发散（>1000）
- ❌ 梯度爆炸/消失

## 故障排除

### 问题：性能不稳定，训练崩溃
**解决方案**: 
- 确认使用了自动超参数配置（不要手动指定）
- 检查是否使用了最新的 `iql.py`（包含状态归一化）
- 尝试降低学习率

### 问题：hopper-medium-expert 性能低于 100
**解决方案**:
- 确认 `temperature=6.0`, `expectile=0.5`（自动配置）
- 运行多个种子（seed 0-4）取平均
- 检查评估时的 `misc/exp_a_mean` 是否合理

### 问题：想要复现 CORL 的确切结果
**解决方案**:
```bash
# 完全遵循 CORL 配置
python run_iql.py --task hopper-medium-expert-v2 --seed 0 --eval_freq 5
python run_iql.py --task hopper-medium-expert-v2 --seed 1 --eval_freq 5
python run_iql.py --task hopper-medium-expert-v2 --seed 2 --eval_freq 5
# 运行多个种子取平均
```
