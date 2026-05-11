# Dog Lab IO Layout

Policy checkpoint: `dog_walk_terrain/2026-05-10_11-20-05/model_21500.pt`  
Network: 373 → 512 → 256 → 128 → 12（actor-only，部署时只用 actor）

---

## 观测向量（373 维）

观测项按 `ObservationsCfg.PolicyCfg` 声明顺序拼接（`concatenate_terms=True`）。

| 索引 | 维度 | 字段 | 说明 |
|------|------|------|------|
| 0–2 | 3 | `base_lin_vel` | 机身线速度，yaw frame，单位 m/s |
| 3–5 | 3 | `base_ang_vel` | 机身角速度，单位 rad/s |
| 6–8 | 3 | `projected_gravity` | 重力向量在机身系的投影（归一化） |
| 9–11 | 3 | `velocity_commands` | 速度指令 [vx, vy, ω_z]，单位 m/s / rad/s |
| 12–23 | 12 | `joint_pos` | 关节角度相对默认站姿的偏差，单位 rad |
| 24–35 | 12 | `joint_vel` | 关节速度，单位 rad/s |
| 36–47 | 12 | `actions` | 上一时刻 policy 输出的动作 |
| 48–372 | 325 | `height_scan` | 高度扫描（见下文），单位 m，clip [-1, 1] |

### 关节顺序（joint_pos / joint_vel / actions，索引 12–47 共用）

```
[0]  FL_hip_joint      [6]  RL_hip_joint
[1]  FR_hip_joint      [7]  RR_hip_joint
[2]  FL_thigh_joint    [8]  RL_thigh_joint
[3]  FR_thigh_joint    [9]  RR_thigh_joint
[4]  FL_calf_joint     [10] RL_calf_joint
[5]  FR_calf_joint     [11] RR_calf_joint
```

### height_scan 网格（obs[48:373]，325 维）

传感器挂载在 `base_link`，`ray_alignment="yaw"`（随偏航旋转），光线投射起点相对 base_link 偏移 (+0.7 m, 0, +20 m)，光线向下，扫描前方地形。

| 参数 | 值 |
|------|----|
| 网格尺寸 | 1.2 m × 0.6 m（传感器本地系） |
| 分辨率 | 0.05 m |
| x 点数 | 25，范围 −0.60 → +0.60 m（传感器系）/ +0.10 → +1.30 m（base_link 系） |
| y 点数 | 13，范围 −0.30 → +0.30 m |
| 总点数 | 25 × 13 = **325**，无 mask |
| 排列顺序 | **x-major**（外循环 y，内循环 x）：`index = y_idx * 25 + x_idx` |
| 训练噪声 | ±0.03 m（部署时不加） |
| clip | [−1.0, +1.0] m |
| 含义 | `sensor_z − hit_z`，正值 = 地面低于传感器（凹陷/平地），负值 = 地面高于传感器（台阶/凸起） |

**坐标换算（base_link 系，机器人朝 +x 方向）：**

```python
# obs[48 + i] 对应的空间坐标
x_base = 0.10 + (i % 25) * 0.05   # 范围 [0.10, 1.30] m，+x = 正前方
y_base = -0.30 + (i // 25) * 0.05 # 范围 [-0.30, +0.30] m，+y = 左侧

# 反过来：已知格子坐标 → 索引
col = round((x_base - 0.10) / 0.05)  # 0..24
row = round((y_base + 0.30) / 0.05)  # 0..12
i   = row * 25 + col
```

**重塑为 2D 网格（可视化 / 卷积输入）：**

```python
grid = obs[48:373].reshape(13, 25)
# grid[row, col]
#   row 0  → y = −0.30 m（右侧）  row 12 → y = +0.30 m（左侧）
#   col 0  → x = +0.10 m（近端）  col 24 → x = +1.30 m（远端）
```

**俯视示意图（机器人在图左，朝 → 方向前进）：**

```
y(m)  近端(x=0.1)  ...  远端(x=1.3)
+0.30 [300][301]...[324]   ← 左侧
+0.25 [275][276]...[299]
 ...
+0.00 [150][151]...[174]   ← 正中
 ...
-0.25 [ 25][ 26]...[ 49]
-0.30 [  0][  1]...[ 24]   ← 右侧
       col0  col1  col24
       ↑ 机器人在此后方，朝右(+x)行进
```

---

## 动作向量（12 维）

`JointPositionActionCfg(use_default_offset=True)`，网络输出为相对默认站姿的位置残差。

实际关节目标：`q_target = q_default + action × scale`

关节顺序与观测相同（见上表）。

### 动作 scale

| 关节组 | scale |
|--------|-------|
| `.*_hip_joint` | 0.15 |
| `.*_thigh_joint` | 0.20 |
| `.*_calf_joint` | 0.15 |

### 默认站姿 q_default（单位 rad，Isaac Lab 仿真空间）

```
FL_hip_joint    =  0.0    FR_hip_joint    =  0.0
RL_hip_joint    =  0.0    RR_hip_joint    =  0.0
FL_thigh_joint  =  0.7    FR_thigh_joint  =  0.7
RL_thigh_joint  =  0.7    RR_thigh_joint  =  0.7
FL_calf_joint   = -1.2    FR_calf_joint   = -1.2
RL_calf_joint   = -1.2    RR_calf_joint   = -1.2
```

### 仿真空间 → URDF 空间的符号映射

URDF 中 FL/RL hip axis=(+1,0,0)，FR/RR hip axis=(-1,0,0)，但 USD 转换后符号丢失，仿真中统一正方向。部署到真机时需乘以以下符号：

```python
DOG_JOINT_SIGN = {
    "FL_hip_joint":  +1.0,   "FR_hip_joint":  -1.0,
    "RL_hip_joint":  -1.0,   "RR_hip_joint":  +1.0,
    # thigh / calf 全为 +1.0
}
```

---

## Critic 观测（仅训练，不部署）

与 Policy 观测完全相同，但 `enable_corruption=False`（无噪声）。
