# DogWalkV2 奖励函数精简记录

**日期**: 2026-05-03  
**分支**: feat/dog-walk-v2  
**关联文档**: `docs/training-lessons-dog-walk-v2.md`

---

## 改动动机

前一轮训练（v2 Phase 1→3）暴露了两类设计问题：

1. **权重偏离规范**：`lin_vel_z_l2`、`flat_orientation_l2`、`ang_vel_xy_l2` 均偏重 2-4 倍，导致机器人在台阶/斜坡上"不敢颠簸不敢倾斜"，平台期卡在 terrain_level ~4.3
2. **补丁项堆积**：`hip_deviation`、`prolonged_air`、`gait_clock` 均为解决具体观察问题而临时添加，导致优化目标重叠、互相干扰

本次改动的原则：**每个奖励项对应唯一一个行为目标，权重贴近规范设计**。

---

## 具体改动

### 1. 默认髋关节站姿（`dog_cfg.py`）

| 关节 | 改前 | 改后 | 原因 |
|------|------|------|------|
| FL_hip_joint | 0.1 rad | 0.0 rad | 中立站姿，policy 无需主动收腿 |
| FR_hip_joint | -0.1 rad | 0.0 rad | 同上 |
| RL_hip_joint | -0.1 rad | 0.0 rad | 同上 |
| RR_hip_joint | 0.1 rad | 0.0 rad | 同上 |

**效果**：`use_default_offset=True` 下 policy 零输出即为竖直站姿，不再需要 `hip_deviation` 惩罚项。

### 2. `DogWalkV2RewardsCfg` 删除项

| 项 | 删除原因 |
|----|----------|
| `hip_deviation` | 默认站姿已为 0，`joint_deviation` 已覆盖偏离惩罚 |
| `prolonged_air` | 与 `feet_air_time` 功能重叠（正负两侧同时约束同一行为） |

### 3. 权重修正到规范值

| 项 | 改前 | 改后 | 规范值 |
|----|------|------|--------|
| `lin_vel_z_l2` | -1.5 | **-0.5** | -0.5 |
| `flat_orientation_l2` | -2.0 | **-0.5** | -0.5 |
| `ang_vel_xy_l2` | -0.2 | **-0.1** | -0.1 |
| `feet_air_time` threshold | 0.35s | **0.5s** | 0.5s |

### 4. `gait_clock` 分阶段策略

- **Phase 1（平地）**：保留 weight=0.5，引导初始 trot 步态形成
- **Phase 1.5 / 1.75 / 2 / 3（地形）**：各子类 `__post_init__` 设 weight=0.1

**原因**：完全置 0 会导致三腿局部最优（一条腿永久悬空，`feet_air_time` 对永久悬空不惩罚）。  
weight=0.1 提供弱约束防止步态退化，但不足以强制固定步频节奏，策略仍可自由适应地形。  
这是行业标准做法：terrain 阶段降权重而非关掉 clock。

---

## 精简后的 Phase 1 奖励全览

| 项 | weight | 行为目标 |
|----|--------|----------|
| `track_lin_vel_xy_exp` | 2.0 | 追踪前向/侧向速度 |
| `track_ang_vel_z_exp` | 0.5 | 追踪转向角速度 |
| `track_height_exp` | 1.5 | 维持目标站高 |
| `feet_air_time` | 0.5 | 激励真实抬腿步态 |
| `foot_slip` | -0.1 | 抑制落脚打滑 |
| `lin_vel_z_l2` | -0.5 | 抑制躯干垂直弹跳 |
| `ang_vel_xy_l2` | -0.1 | 抑制翻滚/俯仰晃动 |
| `flat_orientation_l2` | -0.5 | 鼓励躯干水平 |
| `dof_torques_l2` | -1e-5 | 节能 |
| `dof_acc_l2` | -5e-7 | 抑制关节颤振 |
| `joint_deviation` | -0.005 | 偏离默认姿态惩罚 |
| `gait_clock` | 0.5 | 引导 1.5Hz trot（地形阶段置 0） |
| `undesired_contacts` | -2.0 | 禁止小腿碰地 |

---

## 待验证（训练结果待填入）

### Phase 1 基准目标

训练目标：mean_reward 稳定、time_out 率 > 85%、步态视觉正常

| 日期 | checkpoint | mean_reward | time_out% | 备注 |
|------|------------|-------------|-----------|------|
| 2026-05-03 | `logs/rsl_rl/dog_locomotion_walk_v2/2026-05-03_13-09-24/model_3800.pt` | ~61 | 94.9% | iter 3800，提前停止（已收敛）；stdout: `logs/train_stdout/phase1_v3_20260503_130914.log` |

**结论**：~850 轮 time_out 已超 85%，~3300 轮收敛至 94.9%，奖励权重修正后收敛速度显著快于上一轮。

### Phase 1.5 目标：terrain_levels ≥ 3.0

热启动自：`logs/rsl_rl/dog_locomotion_walk_v2/2026-05-03_13-09-24/model_3800.pt`  
stdout 日志：`logs/train_stdout/phase1p5_v3_20260503_140743.log`

| 日期 | checkpoint | terrain_levels | 备注 |
|------|------------|----------------|------|
| 2026-05-03 | `logs/rsl_rl/dog_locomotion_walk_v2_phase1p5/2026-05-03_15-40-24/model_5000.pt` | ~1.96 | iter ~5000，plateau；stdout: `logs/train_stdout/phase1p5_v3_20260503_145325.log` |

**结论**：terrain_levels 在 1.9 附近收敛，未达 3.0 目标。地形太容易（最大坡 6.9°+起伏 3cm），策略已充分适应，继续训无收益。直接进 Phase 1.75 让台阶难度推动进一步学习。

### Phase 1.75 目标：terrain_levels ≥ 4.0

热启动自：`logs/rsl_rl/dog_locomotion_walk_v2_phase1p5/2026-05-03_15-40-24/model_5000.pt`  
stdout 日志：`logs/train_stdout/phase1p75_v3_20260503_*.log`

| 日期 | checkpoint | terrain_levels | 备注 |
|------|------------|----------------|------|
| — | — | — | 训练中 |

### Phase 2 目标：terrain_levels ≥ 7.0（10cm 台阶、14° 坡度）

| 日期 | checkpoint | terrain_levels | 备注 |
|------|------------|----------------|------|
| — | — | — | 待训练 |

---

---

## V3 大改版（2026-05-03 下午）

### 根因分析

Phase 1.5 训练（v2）出现持续性三腿局部最优（RR 腿永久悬空）。排查发现根因在 Phase 1 阶段就已存在：

1. **`gait_clock` × `feet_air_time` 数学矛盾**
   - `gait_clock` 1.5Hz：周期 0.667s，要求每腿每 0.333s 落地一次
   - `feet_air_time` threshold=0.5s：要求每次腾空至少 0.5s
   - 0.333s < 0.5s → 物理不可能同时满足 → policy 选择每个周期踏地两次（双踏步）以绕过矛盾
   - 观察到的现象：每个步态周期每条腿踏地两次，与正常 trot 完全不同

2. **`feet_air_time` 比例过低**
   - v2 设计：`feet_air_time`=0.5 vs `track_lin_vel_xy_exp`=2.0，比例 1:4
   - RSL-RL 论文基准：两者比例约 1:1
   - 结果：速度奖励压倒步态奖励，三腿滑步在速度上不劣于四腿行走

3. **奖励项目膨胀至 13 项**，多项存在功能重叠（`gait_clock` + `feet_air_time`、`track_height_exp` + `flat_orientation_l2`、`joint_deviation` + 初始化站姿）

### V3 改动

#### 删除项

| 删除项 | 原因 |
|--------|------|
| `gait_clock` | 与 `feet_air_time` 数学矛盾，是三腿问题根源；完全删除 |
| `track_height_exp` | 身高由 `flat_orientation_l2` + 腿部摩擦力隐式约束，重复 |
| `foot_slip` | 权重过小（-0.1）且 `dof_torques_l2` 已惩罚滑脚带来的无效扭矩 |
| `joint_deviation` | 默认站姿已为 0.0（中立），policy 零输出即为合理站姿，惩罚无实质意义 |

#### 权重重新对标 RSL-RL 论文

| 项 | V2 weight | V3 weight | RSL-RL 基准 |
|----|-----------|-----------|-------------|
| `track_lin_vel_xy_exp` | 2.0 | **1.5** | 1.0 |
| `track_ang_vel_z_exp` | 0.5 | **0.75** | 0.5 |
| `feet_air_time` | 0.5 | **1.0** | 1.0 |
| `lin_vel_z_l2` | -0.5 | **-1.0** | -2.0 |
| `flat_orientation_l2` | -0.5 | **-1.0** | -5.0 |
| `ang_vel_xy_l2` | -0.1 | **-0.05** | -0.05 |
| `undesired_contacts` | -2.0 | **-1.0** | — |

#### 精简后 10 项奖励（V3 基准）

| 项 | weight | 行为目标 |
|----|--------|----------|
| `track_lin_vel_xy_exp` | 1.5 | 追踪前向/侧向速度 |
| `track_ang_vel_z_exp` | 0.75 | 追踪转向角速度 |
| `feet_air_time` | 1.0 | 激励真实四腿步态 |
| `lin_vel_z_l2` | -1.0 | 抑制躯干垂直弹跳 |
| `ang_vel_xy_l2` | -0.05 | 抑制翻滚/俯仰晃动 |
| `flat_orientation_l2` | -1.0 | 鼓励躯干水平 |
| `dof_torques_l2` | -1e-5 | 节能 |
| `dof_acc_l2` | -2.5e-7 | 抑制关节颤振 |
| `action_rate_l2` | -0.005 | 动作平滑 |
| `undesired_contacts` | -1.0 | 禁止小腿碰地 |

#### 观测空间变化

- 删除 `gait_phase`（2 维）：obs 维度缩减，与 v2 checkpoint 不兼容，需从头训练
- 删除 `height_command`（原 `UniformVelocityHeightCommandCfg` → `UniformVelocityCommandCfg`）：command 由 3 维降为 2 维

**所有 v2 checkpoint 均无法继续使用，Phase 1 必须从头训练。**

---

## V3 训练结果

### Phase 1 v3 目标

训练目标：mean_reward 稳定、time_out 率 > 85%、四腿步态视觉正常（无三腿局部最优）

| 日期 | checkpoint | mean_reward | time_out% | 备注 |
|------|------------|-------------|-----------|------|
| — | — | — | — | 训练中 |

### Phase 1.5 v3 目标：terrain_levels ≥ 3.0

热启动自：Phase 1 v3 checkpoint（待填入）

| 日期 | checkpoint | terrain_levels | 备注 |
|------|------------|----------------|------|
| — | — | — | 待训练 |

### Phase 1.75 v3 目标：terrain_levels ≥ 4.0

| 日期 | checkpoint | terrain_levels | 备注 |
|------|------------|----------------|------|
| — | — | — | 待训练 |

### Phase 2 v3 目标：terrain_levels ≥ 7.0

| 日期 | checkpoint | terrain_levels | 备注 |
|------|------------|----------------|------|
| — | — | — | 待训练 |

---

## 后续迭代方向

1. 若 Phase 1 步态仍出现三腿局部最优，根因应在 `feet_air_time` 比例，可进一步提高至 1.5；不要恢复 `gait_clock`
2. action scale（thigh/calf）当前为 0.20/0.25，对应最大抬脚约 9.5cm；如需通 10cm 台阶应升至 0.25/0.30，在 Phase 1 稳定后调整（已在 Phase 3 中启用）
3. Phase 3 的 `lin_vel_z_l2=-0.3`、`flat_orientation_l2=-0.8` 为松弛值（基准 -1.0），允许身体起伏适应台阶
