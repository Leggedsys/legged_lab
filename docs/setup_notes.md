# 训练记录

## URDF → USD 轴符号问题

左右 hip 关节轴方向相反（FL/RL axis=1, FR/RR axis=-1），但 URDF→USD 转换时 axis 符号丢失。
所有 hip 在 USD 中 axis 都变成正方向，所以 `init_state` 中 FR/RR hip 需取反才能正确外摆。

部署到真机时需通过 `DOG_JOINT_SIGN` 映射还原 URDF 空间的值。

## 关键参数修改记录

### dog_cfg.py

| 参数 | 修改 | 原因 |
|---|---|---|
| `max_depenetration_velocity` | 1.0 → 3.0 | 防止脚掌陷地 |
| `solver_position_iteration_count` | 4 → 8 | 改善接触稳定性 |
| `init_state.joint_pos.hip` | 0.0 → 0.2 → 0.1 | 0.0 站姿不稳易摔；0.2 略宽，改 0.1 折中 |
| `stiffness` | 20 | DCMotor，站立够用 |

### dog_env_cfg.py (DogEnvCfg — 站立任务)

| 参数 | 修改 | 原因 |
|---|---|---|
| `thigh action scale` | 0.35 → 0.2 | 配合限位 [-1.6, 0.8]，默认 0.7 + 0.2 = 0.9 不撞限 |
| `alive.weight` | 0.5 → 1.0 | 提高存活 incentive |
| `joint_deviation.weight` | -0.2 → -0.5 → -0.1 | 初始 -0.2，后改 -0.5 太重导致负 reward，回退到 -0.1 |

### dog_env_cfg.py (DogWalkEnvCfg — 行走任务)

| 参数 | 值 | 原因 |
|---|---|---|
| `lin_vel_x` | (-0.3, 0.6) | 前后行走 |
| `lin_vel_y` | (-0.2, 0.2) | 侧移 |
| `ang_vel_z` | (-0.5, 0.5) | 转向 |
| `rel_standing_envs` | 0.1 | 10% 站立，防退化 |
| `resampling_time_range` | (4.0, 8.0) | 指令每 4~8s 重采样 |
| `alive.weight` | 0.0 → 0.5 | 原为 0，策略从零探索时摔倒 → 学废不动；加 0.5 给安全底，鼓励试探 |
| `track_lin_vel_xy_exp.weight` | 3.5 | 主力 reward |
| `track_ang_vel_z_exp.weight` | 1.0 | 转向 reward |
| `tracking std` | 0.5 → 0.25 | 原值太大，站着不动就拿 70%；缩到 0.25 后只拿 24%，迫使学走路 |
| `feet_air_time.weight` | 0.5 | 鼓励抬腿形成步态 |

### PPO 配置

| 参数 | 值 |
|---|---|
| `max_iterations` | 800 |
| `num_steps_per_env` | 32 |
| `actor_hidden_dims` | [512, 256, 128] |
| `learning_rate` | 3e-4 |
| `init_noise_std` | 0.2 (行走用 0.05) | 行走任务初始噪声从 0.2 降到 0.05，减少早期随机摔倒 |
| `entropy_coef` | 0.001 |
| `empirical_normalization` | False |

### Episode 参数

- `dt=0.005`, `decimation=4` → 控制周期 20ms
- `episode_length_s=10.0` → 每局 500 步
- `num_envs=128`

### ObservationsCfg — 修复

原来缺少 `base_lin_vel`，导致策略看不到自己走没走，无法形成 velocity tracking 的闭环反馈。
加了之后网络输入多了 3 维（body-frame vx, vy, vz）。

### feet_air_time 函数 — 修复

原来用的是 `feet_air_time_positive_biped`（双足专用），只在单脚离地时给 reward。
四足行走时通常 2-3 脚着地，这个 reward 几乎从不会被激活。
改为通用 `feet_air_time`，对所有脚独立计算离地时间。

### 与官方配置（Go2/Spot/ANYmal）对比总结

| 项目 | 官方 | 本配置 | 影响 |
|---|---|---|---|
| `base_lin_vel` 观测 | ✅ 有 | ❌ 缺（已补） | **致命**，无速度反馈学不会跟踪 |
| `feet_air_time` 函数 | `feet_air_time` | `feet_air_time_positive_biped`（已改） | **重要**，双足函数对四足几乎不激活 |
| `track_lin_vel_xy` 函数 | `track_lin_vel_xy_exp` | `track_lin_vel_xy_yaw_frame_exp` | 两者都是 yaw frame，等效 |
| `track_ang_vel_z` 函数 | `track_ang_vel_z_exp` | `track_ang_vel_z_world_exp` | 等效 |
| `lin_vel_z_l2` weight | -2.0 | -0.5 | 更轻，合适 |
| `ang_vel_xy_l2` weight | -0.05 | -0.15 | 略重但不致命 |
| `dof_torques_l2` weight | -1.0e-5 | -2.0e-5 | 略重 |
| `action_rate_l2` weight | -0.01 | -0.005 | 更轻 |
| `undesired_contacts` body | `.*THIGH` | `.*_calf` | 用户选择，合理 |
| `flat_orientation_l2` | 0.0（关） | -1.0（开） | 用户选择 |
| `reset_joints_by_scale` 范围 | Go2: (1.0, 1.0)不随机 | (0.9, 1.1) | 略微随机 |
| `reset_base` 速度范围 | ±0.5 | 0.0 | 更保守 |
| `push_robot` 扰动事件 | 有 | 无 | 无外力扰动 |
| `base_com` 质心随机 | 有 | 无 | 无质心偏移 |
| `height_scan` 观测 | 有（地形用） | 无（平地不用） | 平地不需要 |
| `rel_standing_envs` | 0.02 | 0.1 | 略多，合理 |

## 已知问题

- DOG_JOINT_SIGN 目前只作为参考字典，未在训练代码中自动使用。部署时需手动映射。
