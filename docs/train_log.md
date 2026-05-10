# 三腿问题的修正
目前的奖励函数结构：
```python
@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.2,
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 1.0,
        },
    )
```

## 怀疑是 dof_acc_l2 和 action_rate_l2 重叠导致的
两项都在约束"不要动太猛"，位置控制下高度相关。保留 action_rate（直接对应真机控制信号），去掉 dof_acc，减少一个需要调权的维度。

实验结果：腿抬高的程度有所缓解，但仍然悬空。

## 加入功率分布惩罚，让电机功率分配更平均作为对称约束
```python
def joint_power_distribution(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    惩罚各关节功率分布不均匀。
    功率 P_i = |τ_i| * |θ̇_i|，惩罚 var(P) 的平方。
    
    三条腿步态时：悬空腿 P≈0，支撑腿 P 极大 → 方差高 → 惩罚大。
    四条腿均匀步态：各腿功率接近 → 方差小 → 惩罚小。
    """
    # shape: (num_envs, num_joints)
    torques = env.scene["robot"].data.applied_torque
    joint_vel = env.scene["robot"].data.joint_vel

    power = torch.abs(torques) * torch.abs(joint_vel)          # (N, 12)
    power_var = torch.var(power, dim=1)                         # (N,)
    return power_var ** 2
```
```python
joint_power_distribution = RewTerm(
    func=mdp.joint_power_distribution,
    weight=-1e-5,
)
```
实验效果：有改善，但仍未根治

## 尝试切换初始化随机种子
训练到600轮以后似乎没有右后腿卷曲的现象了，目前暂时认为问题解决了
在地形训练中轮数加大，RR抬腿问题又再次出现

## 尝试加prolonged_air_penalty来惩罚长时间悬空
```python
prolonged_air_penalty = RewTerm(
    func=mdp.prolonged_air_penalty,
    weight=-1.0,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "threshold": 0.5,
    },
)
```

## 怀疑是feet_air_time的问题，将其去掉
由于feet_air_time是触底后抬脚，然后计算悬空的累计奖励，策略学会使用一只脚一直悬空，然后累积足够奖励后触地收割的技巧。

但还是无法解决。

---

## 根本解法：终止条件而非奖励惩罚（2026-05-09）

**结论：所有基于方差/连续惩罚的方法（foot_contact_balance、joint_power_distribution、prolonged_air_penalty）均无法根治三腿问题。根本原因是速度跟踪奖励（weight=1.5）的梯度远强于惩罚项，策略始终选择"忍受惩罚换速度"。**

### 有效方案：`foot_prolonged_air_termination`

将三腿判断从奖励项改为**终止条件**：任意一只脚连续离地超过阈值秒数，则立即终止该回合。

```python
# dog_env_cfg.py — TerminationsCfg
prolonged_air = DoneTerm(
    func=mdp.foot_prolonged_air_termination,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "threshold": 1.0,  # 初始设 1.5s，后调整为 1.0s
    },
)
```

```python
# mdp/rewards.py
def foot_prolonged_air_termination(env, sensor_cfg, threshold=1.0):
    sensor = env.scene[sensor_cfg.name]
    current_air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return (current_air_time > threshold).any(dim=1)  # bool (N,)
```

**为什么终止有效而惩罚无效：**
- 惩罚是连续梯度信号，与速度收益对抗，策略会在"三腿稍贵但速度好"和"四腿稍慢但无惩罚"之间优化，结果取决于权重比例，很难调到彻底消除
- 终止是离散事件，直接截断后续所有未来收益，代价不可补偿；策略学到"三腿 = 死路"，而非"三腿 = 贵"

**训练结果（5000 iter flat terrain，4096 envs）：**
- `bad_orientation`：72.5% → 4.4%（姿态稳定性极大改善）
- `prolonged_air` 终止率：稳定在 ~12%（终止条件正常工作）
- 视觉验证（model_3400.pt GUI 播放）：三腿问题几乎不可见

**阈值选择：**
- 正常 trot 摆腿约 0.2–0.3s，threshold=1.0s 不会误触发
- 初始使用 1.5s，后收紧到 1.0s 以减少 12% 的终止率

---

## 里程碑 M1：地形课程突破 level 3.8（2026-05-10）

### 背景

在 flat terrain 完成预训练（~17500 iter）后切换至 `COMPETITION_TERRAIN_IMPORTER_CFG`（10 行 × 20 列，10 级课程）。首轮地形训练（±10cm 高度扫描噪声）卡在 **terrain_levels ≈ 3.7** 无法突破。

### 根因：高度扫描噪声掩盖地形信号

| Level | 台阶高度 | ±10cm 噪声 SNR | ±3cm 噪声 SNR |
|---|---|---|---|
| 3 | 3.7cm | **0.37×** | 1.2× |
| 4 | 4.4cm | **0.44×** | 1.5× |
| 5 | 5.5cm | **0.55×** | 1.8× |

±10cm 的噪声完全掩盖了 level 3–5 的台阶高度（SNR < 1），策略在训练时感知不到地形结构，只能靠平地步态盲走，遇台阶跌倒，课程无法推进。

Play 模式（无噪声）下机器人可轻松通过 level 10，验证策略能力没问题——是感知噪声导致的训练瓶颈，不是策略表达能力问题。

### 修复

将 PolicyCfg 高度扫描噪声从 ±10cm 降至 ±3cm（匹配真机 D435 深度相机实际精度）：

```python
# dog_env_cfg.py — ObservationsCfg.PolicyCfg
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
    noise=Unoise(n_min=-0.03, n_max=0.03),  # 原 -0.1, 0.1
    clip=(-1.0, 1.0),
)
```

Critic 保持无噪声（`enable_corruption=False`），不受影响，可直接 resume。

### 训练曲线（resume 自 iter 17500）

| iter 区间 | terrain_levels | 说明 |
|---|---|---|
| 17500–18500 | 1.7 → 3.6 | 噪声修复生效，快速上升 |
| 18500–19800 | 3.6 → 3.8 | 持续推进 |
| 19800–21736 | **3.71–3.84 震荡** | 新平台，触发自动终止 |

**终止时指标（iter 21736）：**
- terrain_levels：3.77 | Mean reward：26–28
- time_out：81% | root_height：10% | prolonged_air：6%
- error_vel_xy：0.28 m/s | error_vel_yaw：0.47 rad/s

### 为什么卡在 level 3.8

1. **SNR 仍不足**：±3cm 对 level 4–5（4.4–5.5cm 台阶）SNR 只有 1.5–1.8×，感知处于临界区
2. **`foot_clearance_reward` 使用世界坐标 Z**：台阶上脚的 Z 值升高 → 惩罚反而减小 → 台阶上没有抬脚激励，信号方向相反
3. **Level 4→5 难度陡增**：台阶高度从 4.4cm 涨到 5.5cm（+25%），level 5 推进率骤降，形成均值稳定在 3.8 的稳态

### 当前最优 Checkpoint

```
logs/rsl_rl/dog_walk_terrain/2026-05-10_11-20-05/
  model_21500.pt   ← 最优权重
  exported/
    policy.pt      ← JIT 导出，可直接部署
    policy.onnx
```

### 下一步计划

1. **降低高度扫描噪声至 ±1cm**（SNR 升至 4.4×@L4），Resume 训练，预期突破 level 5
2. **修复 `foot_clearance_reward`**：改为地形相对高度，或换用 `foot_air_time` 激励抬脚

### 工程修复（本里程碑同期）

| 问题 | 修复 |
|---|---|
| `train.py` / `play.py` log 路径相对工作目录，日志散落在 `scripts/rsl_rl/logs/` | 改为相对 `__file__` 锚定项目根，统一写入 `./logs/` |
| `RayCasterCfg` 使用弃用参数 `attach_yaw_only` | 改为 `ray_alignment="yaw"`，消除每步刷屏警告 |
| `UniformVelocityCommandCfg` 多余的 `heading=(0.0, 0.0)` 配置 | 删去，消除启动警告 |
| play 时所有机器人堆在低等级地形 | `max_init_terrain_level` play 时设为 9，训练时保持 1 |