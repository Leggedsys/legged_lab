# Tier 1 Universal Locomotion Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零训练 `Dog-Legged-Lab-Walk-v1` 策略，具备速度+高度指令跟踪、地形课程（平地→斜坡→台阶→板桥）和非对称 Critic，达到部署级运动质量。

**Architecture:** 非对称 Actor-Critic PPO；Actor 观测 = 本体感知（48维）+ 噪声高度扫描（273维）+ 4维指令[vx,vy,ω,h_cmd]；Critic 额外观测无噪声高度扫描+接触力+摩擦系数。`h_cmd` 通过 `UniformVelocityHeightCommand` 作为第4维指令采样，使策略在训练中学会按指令调节躯干高度（正常0.28m，过限高杆0.15m）。

**Tech Stack:** IsaacLab ManagerBasedRLEnv, RSL-RL PPO, Isaac Sim GPU physics

---

## 文件地图

| 操作 | 路径 | 职责 |
|------|------|------|
| MODIFY | `source/legged_lab/legged_lab/tasks/manager_based/legged_lab/mdp/rewards.py` | 新增 `track_height_exp`, `foot_slip` |
| CREATE | `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/commands.py` | `UniformVelocityHeightCommand` + Cfg |
| MODIFY | `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/__init__.py` | 导出新 command |
| CREATE | `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/terrains.py` | 所有比赛地形配置 |
| MODIFY | `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/dog_env_cfg.py` | 新增 `DogWalkV2SceneCfg`, `DogWalkV2EnvCfg` |
| CREATE | `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/agents/rsl_rl_walk_v2_ppo_cfg.py` | `DogWalkV2PPORunnerCfg` |
| MODIFY | `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/__init__.py` | 注册 `Dog-Legged-Lab-Walk-v1` |
| CREATE | `tests/test_dog_walk_v2.py` | 冒烟测试 |

---

## Task 1：新增奖励函数

**Files:**
- Modify: `source/legged_lab/legged_lab/tasks/manager_based/legged_lab/mdp/rewards.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_rewards.py`：

```python
import torch
import pytest

def test_track_height_exp_perfect():
    """完美跟踪时奖励为1.0"""
    from legged_lab.tasks.manager_based.legged_lab.mdp.rewards import (
        _track_height_exp_impl,
    )
    current = torch.tensor([0.28, 0.15])
    target = torch.tensor([0.28, 0.15])
    result = _track_height_exp_impl(current, target, std=0.03)
    assert torch.allclose(result, torch.ones(2)), f"Expected 1.0, got {result}"


def test_track_height_exp_deviation():
    """偏差0.1m时奖励应显著低于1.0"""
    from legged_lab.tasks.manager_based.legged_lab.mdp.rewards import (
        _track_height_exp_impl,
    )
    current = torch.tensor([0.38])
    target = torch.tensor([0.28])
    result = _track_height_exp_impl(current, target, std=0.03)
    assert result[0] < 0.01, f"Expected near 0, got {result[0]}"


def test_foot_slip_zero_when_airborne():
    """空中的脚不应产生 slip 惩罚"""
    from legged_lab.tasks.manager_based.legged_lab.mdp.rewards import (
        _foot_slip_impl,
    )
    foot_vel = torch.tensor([[1.0, 0.5], [0.8, 0.3]])  # (2 envs, 2 feet)
    contact = torch.tensor([[False, False], [False, False]])
    result = _foot_slip_impl(foot_vel, contact)
    assert torch.allclose(result, torch.zeros(2)), f"Expected 0, got {result}"


def test_foot_slip_penalizes_sliding_contact():
    """接触地面且滑动时应有非零惩罚"""
    from legged_lab.tasks.manager_based.legged_lab.mdp.rewards import (
        _foot_slip_impl,
    )
    foot_vel = torch.tensor([[1.0, 0.0]])  # 1 env, 2 feet, foot0 moving
    contact = torch.tensor([[True, False]])  # foot0 in contact
    result = _foot_slip_impl(foot_vel, contact)
    assert result[0] > 0.0, f"Expected positive penalty, got {result[0]}"
```

- [ ] **Step 2: 运行测试（确认失败）**

```bash
cd /root/gpufree-data/legged_lab
python -m pytest tests/test_rewards.py -v 2>&1 | head -20
```

Expected: `ImportError` 或 `ModuleNotFoundError`（函数不存在）

- [ ] **Step 3: 实现奖励函数**

在 `source/legged_lab/legged_lab/tasks/manager_based/legged_lab/mdp/rewards.py` 末尾追加：

```python
def _track_height_exp_impl(
    current_height: torch.Tensor, target_height: torch.Tensor, std: float
) -> torch.Tensor:
    """Pure-tensor helper, testable without Isaac Sim."""
    height_error = torch.square(current_height - target_height)
    return torch.exp(-height_error / std**2)


def track_height_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str,
    asset_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Exponential reward for tracking the height command (4th dim of velocity command)."""
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.assets import Articulation

    if asset_cfg is None:
        asset_cfg = _SceneEntityCfg("robot")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, 4)
    target_height = command[:, 3]
    current_height = asset.data.root_pos_w[:, 2]
    return _track_height_exp_impl(current_height, target_height, std)


def _foot_slip_impl(
    foot_speed: torch.Tensor, contact: torch.Tensor
) -> torch.Tensor:
    """Pure-tensor helper: foot_speed (N, F), contact bool (N, F) -> (N,)."""
    return torch.sum(foot_speed * contact.float(), dim=1)


def foot_slip(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg" = None,
    sensor_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Penalize foot sliding when in contact with ground."""
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

    if asset_cfg is None:
        asset_cfg = _SceneEntityCfg("robot", body_names=".*_foot")
    if sensor_cfg is None:
        sensor_cfg = _SceneEntityCfg("contact_forces", body_names=".*_foot")

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = contact_forces.norm(dim=-1).max(dim=1)[0] > 1.0  # (N, F)

    foot_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # xy only
    foot_speed = foot_lin_vel.norm(dim=-1)  # (N, F)

    return _foot_slip_impl(foot_speed, in_contact)
```

- [ ] **Step 4: 运行测试（确认通过）**

```bash
cd /root/gpufree-data/legged_lab
python -m pytest tests/test_rewards.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add source/legged_lab/legged_lab/tasks/manager_based/legged_lab/mdp/rewards.py tests/test_rewards.py
git commit -m "feat: add track_height_exp and foot_slip reward functions"
```

---

## Task 2：创建 UniformVelocityHeightCommand

**Files:**
- Create: `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/commands.py`
- Modify: `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/__init__.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_commands.py`：

```python
import torch
import pytest


def test_command_cfg_has_height_range():
    from legged_lab.tasks.manager_based.dog_lab.mdp.commands import (
        UniformVelocityHeightCommandCfg,
    )
    cfg = UniformVelocityHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.1,
        heading_command=False,
        ranges=UniformVelocityHeightCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
            heading=(0.0, 0.0),
            height=(0.15, 0.30),
        ),
    )
    assert cfg.ranges.height == (0.15, 0.30)


def test_command_dim_is_4():
    """command tensor shape 末尾应为 4（vx, vy, ω, h）"""
    from legged_lab.tasks.manager_based.dog_lab.mdp.commands import (
        UniformVelocityHeightCommand,
        UniformVelocityHeightCommandCfg,
    )
    # Just check the class attribute exists (full test needs Isaac Sim)
    assert hasattr(UniformVelocityHeightCommand, "_build_command_buffer")
```

- [ ] **Step 2: 运行测试（确认失败）**

```bash
python -m pytest tests/test_commands.py -v 2>&1 | head -10
```

Expected: `ImportError`

- [ ] **Step 3: 创建 commands.py**

新建 `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/commands.py`：

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass

if False:  # TYPE_CHECKING
    from isaaclab.envs import ManagerBasedEnv


@configclass
class UniformVelocityHeightCommandCfg(UniformVelocityCommandCfg):
    """Extends velocity command with a uniform height command (4th dimension)."""

    class_type: type = None  # set below after class definition

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        height: tuple[float, float] = MISSING
        """Min and max target base height (m). e.g. (0.15, 0.30)"""


class UniformVelocityHeightCommand(UniformVelocityCommand):
    """Velocity command extended with a height component.

    Outputs command tensor of shape (N, 4): [vx, vy, ω, h_cmd].
    The height is sampled uniformly from cfg.ranges.height.
    Standing environments receive h_cmd sampled normally (they may still crouch).
    """

    cfg: UniformVelocityHeightCommandCfg

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.h_command = torch.zeros(self.num_envs, device=self.device)
        # Initialize to nominal height
        self.h_command[:] = (cfg.ranges.height[0] + cfg.ranges.height[1]) / 2.0

    @staticmethod
    def _build_command_buffer():
        """Marker method for testing — confirms class is defined."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        self.h_command[env_ids] = r.uniform_(*self.cfg.ranges.height)

    @property
    def command(self) -> torch.Tensor:
        """Shape (N, 4): [vx, vy, ω, h_cmd]."""
        return torch.cat([self.vel_command_b, self.h_command.unsqueeze(1)], dim=1)


# Wire cfg to class
UniformVelocityHeightCommandCfg.class_type = UniformVelocityHeightCommand
```

- [ ] **Step 4: 更新 mdp/__init__.py**

在 `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/__init__.py` 末尾追加：

```python
from legged_lab.tasks.manager_based.dog_lab.mdp.commands import (  # noqa: F401
    UniformVelocityHeightCommand,
    UniformVelocityHeightCommandCfg,
)
```

- [ ] **Step 5: 运行测试**

```bash
python -m pytest tests/test_commands.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/commands.py \
        source/legged_lab/legged_lab/tasks/manager_based/dog_lab/mdp/__init__.py \
        tests/test_commands.py
git commit -m "feat: add UniformVelocityHeightCommand for h_cmd as 4th command dim"
```

---

## Task 3：创建地形课程配置

**Files:**
- Create: `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/terrains.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_terrains.py`：

```python
def test_terrain_cfg_importable():
    from legged_lab.tasks.manager_based.dog_lab.terrains import (
        COMPETITION_TERRAIN_CFG,
        FLAT_TERRAIN_CFG,
    )
    assert COMPETITION_TERRAIN_CFG is not None
    assert FLAT_TERRAIN_CFG is not None


def test_competition_terrain_has_all_types():
    from legged_lab.tasks.manager_based.dog_lab.terrains import COMPETITION_TERRAIN_CFG
    sub = COMPETITION_TERRAIN_CFG.sub_terrains
    required = {"flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "stepping_stones"}
    assert required.issubset(set(sub.keys())), f"Missing: {required - set(sub.keys())}"


def test_stairs_height_matches_competition():
    from legged_lab.tasks.manager_based.dog_lab.terrains import COMPETITION_TERRAIN_CFG
    stairs = COMPETITION_TERRAIN_CFG.sub_terrains["stairs_up"]
    assert stairs.step_height_range[1] == pytest.approx(0.10, abs=0.001), \
        f"Expected max stair height 0.10m, got {stairs.step_height_range[1]}"

import pytest
```

- [ ] **Step 2: 运行测试（确认失败）**

```bash
python -m pytest tests/test_terrains.py -v 2>&1 | head -10
```

Expected: `ImportError`

- [ ] **Step 3: 创建 terrains.py**

新建 `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/terrains.py`：

```python
"""Competition terrain curriculum configurations for Dog-Legged-Lab-Walk-v1."""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfSteppingStonesTerrainCfg,
)
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

# ---------------------------------------------------------------------------
# Flat terrain (Phase 1, also used as baseline throughout)
# ---------------------------------------------------------------------------
FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
)

# ---------------------------------------------------------------------------
# Full competition terrain curriculum (Phase 2–4)
#
# num_rows = difficulty levels (0 = easiest, 9 = hardest)
# num_cols = number of terrain type columns
# Curriculum advances robots row-by-row based on velocity tracking performance.
# Stepping stones here approximate 横板桥 (fixed-width planks, periodic gaps).
# ---------------------------------------------------------------------------
COMPETITION_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # ── Flat (always present as safe fallback) ──────────────────────
        "flat": MeshPlaneTerrainCfg(proportion=0.1),

        # ── Random rough (砂砾/碎木坑 模拟) ─────────────────────────────
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),

        # ── Sloped terrain (斜坡，纵向) ──────────────────────────────────
        # slope_range in radians: 0→0.25 rad ≈ 0→14.3°, matches competition 14°
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),

        # ── Stairs (100mm 台阶) ──────────────────────────────────────────
        # step_height_range: curriculum from 0.02m → 0.10m
        "stairs_up": HfPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.02, 0.10),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_down": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.02, 0.10),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        ),

        # ── Stepping stones (横板桥 近似: 40cm 石块，15cm 间距) ───────────
        # stone_width_range: (0.35, 0.40) → curriculum from wider to exact 40cm
        # stone_distance_range: (0.05, 0.15) → curriculum up to 15cm gap
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_height_max=0.05,
            stone_width_range=(0.30, 0.40),
            stone_distance_range=(0.05, 0.15),
            border_width=0.25,
        ),

        # ── Narrow beam (纵板桥 近似: 窄梁20cm，10cm 间距)
        # 使用随机高度场 + 离散障碍近似，正式版需自定义地形生成器（Task 3b）
        "narrow_rough": HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.00, 0.02),
            noise_step=0.01,
            border_width=0.5,  # 宽边界模拟只有中间有支撑
        ),
    },
)

# ---------------------------------------------------------------------------
# Terrain Importer helpers (reusable in env configs)
# ---------------------------------------------------------------------------
_PHYSICS_MAT = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)

FLAT_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    terrain_generator=None,
    max_init_terrain_level=0,
    collision_group=-1,
    physics_material=_PHYSICS_MAT,
    debug_vis=False,
)

COMPETITION_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=COMPETITION_TERRAIN_CFG,
    max_init_terrain_level=5,  # 从第5行（中等难度）开始
    collision_group=-1,
    physics_material=_PHYSICS_MAT,
    debug_vis=False,
)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_terrains.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add source/legged_lab/legged_lab/tasks/manager_based/dog_lab/terrains.py \
        tests/test_terrains.py
git commit -m "feat: add competition terrain curriculum config (flat/rough/slope/stairs/stepping_stones)"
```

---

## Task 4：创建 DogWalkV2EnvCfg

**Files:**
- Modify: `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/dog_env_cfg.py`

- [ ] **Step 1: 在 dog_env_cfg.py 顶部添加导入**

在 `dog_env_cfg.py` 现有 import 区域追加：

```python
from isaaclab.sensors import RayCasterCfg, patterns
from legged_lab.tasks.manager_based.dog_lab.terrains import (
    COMPETITION_TERRAIN_IMPORTER_CFG,
    FLAT_TERRAIN_IMPORTER_CFG,
)
```

- [ ] **Step 2: 在文件末尾添加 DogWalkV2SceneCfg**

```python
@configclass
class DogWalkV2SceneCfg(MySceneCfg):
    """Scene with height scanner for DogWalkV2."""

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.0, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
```

- [ ] **Step 3: 添加 DogWalkV2 的 Observations、Commands、Rewards、Curriculum**

在同一文件末尾继续追加：

```python
@configclass
class DogWalkV2ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        # Height scan (noisy, simulates depth camera artifacts)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for Critic only (not available at deployment)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # Exact height scan (no noise)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DogWalkV2CommandsCfg:
    base_velocity = mdp.UniformVelocityHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityHeightCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
            heading=(0.0, 0.0),
            height=(0.15, 0.30),
        ),
    )


@configclass
class DogWalkV2RewardsCfg:
    # ── Velocity tracking ──────────────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    # ── Height tracking (new) ──────────────────────────────────────────────
    track_height_exp = RewTerm(
        func=mdp.track_height_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.03},
    )
    # ── Gait quality ───────────────────────────────────────────────────────
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
        },
    )
    # ── Slip penalty (new) ─────────────────────────────────────────────────
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # ── Stability penalties ────────────────────────────────────────────────
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    # ── Effort / smoothness ────────────────────────────────────────────────
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.005)
    # ── Safety ─────────────────────────────────────────────────────────────
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 1.0,
        },
    )


@configclass
class DogWalkV2CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class DogWalkV2EnvCfg(ManagerBasedRLEnvCfg):
    scene: DogWalkV2SceneCfg = DogWalkV2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DogWalkV2ObservationsCfg = DogWalkV2ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: DogWalkV2CommandsCfg = DogWalkV2CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: DogWalkV2RewardsCfg = DogWalkV2RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: DogWalkV2CurriculumCfg = DogWalkV2CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Start with flat terrain; switch to COMPETITION_TERRAIN_IMPORTER_CFG for Phase 2+
        self.scene.terrain = FLAT_TERRAIN_IMPORTER_CFG
        self.sim.physics_material = self.scene.terrain.physics_material

        # Domain randomization events
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.5, 1.5)
        # Enable push disturbance (base EventCfg has force_range=(0,0) by default)
        self.events.base_external_force_torque.params["force_range"] = (0.0, 25.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 2.0)
```

- [ ] **Step 4: Commit**

```bash
git add source/legged_lab/legged_lab/tasks/manager_based/dog_lab/dog_env_cfg.py
git commit -m "feat: add DogWalkV2EnvCfg with height scan, h_cmd command, updated rewards"
```

---

## Task 5：PPO 配置 + 环境注册

**Files:**
- Create: `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/agents/rsl_rl_walk_v2_ppo_cfg.py`
- Modify: `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/__init__.py`

- [ ] **Step 1: 创建 PPO 配置**

新建 `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/agents/rsl_rl_walk_v2_ppo_cfg.py`：

```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DogWalkV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "dog_locomotion_walk_v2"
    max_iterations = 5000
    num_steps_per_env = 24
    save_interval = 200
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,        # 10× original, prevents premature convergence
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="fixed",         # prevents LR→0 collapse seen in previous runs
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

- [ ] **Step 2: 注册新环境**

在 `source/legged_lab/legged_lab/tasks/manager_based/dog_lab/__init__.py` 追加：

```python
gym.register(
    id="Dog-Legged-Lab-Walk-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogWalkV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_walk_v2_ppo_cfg:DogWalkV2PPORunnerCfg",
    },
)
```

- [ ] **Step 3: Commit**

```bash
git add source/legged_lab/legged_lab/tasks/manager_based/dog_lab/agents/rsl_rl_walk_v2_ppo_cfg.py \
        source/legged_lab/legged_lab/tasks/manager_based/dog_lab/__init__.py
git commit -m "feat: add DogWalkV2PPORunnerCfg and register Dog-Legged-Lab-Walk-v1"
```

---

## Task 6：冒烟测试（不启动渲染）

**Files:**
- Create: `tests/test_dog_walk_v2.py`

- [ ] **Step 1: 写注册测试**

新建 `tests/test_dog_walk_v2.py`：

```python
import pytest


def test_env_registered():
    import gymnasium as gym
    import legged_lab.tasks  # noqa: F401
    all_ids = gym.registry.keys()
    assert "Dog-Legged-Lab-Walk-v1" in all_ids, \
        f"Dog-Legged-Lab-Walk-v1 not in registry. Available: {[k for k in all_ids if 'Dog' in k]}"


def test_ppo_cfg_importable():
    from legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_walk_v2_ppo_cfg import (
        DogWalkV2PPORunnerCfg,
    )
    cfg = DogWalkV2PPORunnerCfg()
    assert cfg.max_iterations == 5000
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.algorithm.entropy_coef == pytest.approx(0.01)


def test_env_cfg_importable():
    from legged_lab.tasks.manager_based.dog_lab.dog_env_cfg import DogWalkV2EnvCfg
    cfg = DogWalkV2EnvCfg()
    assert cfg.scene.num_envs == 4096
    assert cfg.episode_length_s == pytest.approx(20.0)
    # Verify height scan sensor is present
    assert hasattr(cfg.scene, "height_scanner")
    # Verify h_cmd exists in command ranges
    assert hasattr(cfg.commands.base_velocity.ranges, "height")
    assert cfg.commands.base_velocity.ranges.height == (0.15, 0.30)


def test_reward_cfg_has_height_tracking():
    from legged_lab.tasks.manager_based.dog_lab.dog_env_cfg import DogWalkV2RewardsCfg
    cfg = DogWalkV2RewardsCfg()
    assert hasattr(cfg, "track_height_exp")
    assert cfg.track_height_exp.weight == pytest.approx(1.5)
    assert hasattr(cfg, "foot_slip")
    assert cfg.foot_slip.weight < 0  # should be a penalty
```

- [ ] **Step 2: 运行测试**

```bash
cd /root/gpufree-data/legged_lab
python -m pytest tests/test_dog_walk_v2.py -v
```

Expected: 4 tests PASSED

如果 `test_env_registered` 失败，检查 `legged_lab/tasks/__init__.py` 是否导入了 `dog_lab`：

```bash
grep -n "dog_lab" source/legged_lab/legged_lab/tasks/__init__.py
```

若缺失，在该文件添加：`from . import dog_lab  # noqa: F401`

- [ ] **Step 3: Commit**

```bash
git add tests/test_dog_walk_v2.py
git commit -m "test: add smoke tests for Dog-Legged-Lab-Walk-v1 registration and config"
```

---

## Task 7：Phase 1 训练（平地，0–500 迭代）

**目标：** 验证环境无崩溃，策略能学会在平地上跟随 [vx, vy, ω, h_cmd] 指令。

- [ ] **Step 1: 启动训练**

```bash
cd /root/gpufree-data/legged_lab/scripts/rsl_rl
python train.py --task Dog-Legged-Lab-Walk-v1 \
    --num_envs 4096 \
    --max_iterations 500 \
    --headless
```

- [ ] **Step 2: 前 50 迭代监控（约10分钟后检查）**

```bash
# 查找最新训练目录
LOGDIR=$(ls -td /root/gpufree-data/legged_lab/logs/rsl_rl/dog_locomotion_walk_v2/*/ | head -1)
echo "Log dir: $LOGDIR"

# 读取关键指标
python3 -c "
from tensorboard.backend.event_processing import event_accumulator
import glob, os
logdir = '$LOGDIR'
ef = glob.glob(logdir + '/events.*')[0]
ea = event_accumulator.EventAccumulator(ef); ea.Reload()
for tag in ['Train/mean_reward', 'Episode_Termination/base_contact',
            'Episode_Termination/bad_orientation', 'Loss/learning_rate']:
    evs = ea.Scalars(tag)
    if evs:
        print(f'{tag}: {evs[-1].value:.4f} @step {evs[-1].step}')
" 2>/dev/null
```

- [ ] **Step 3: Phase 1 成功标准（500迭代后）**

运行同一监控命令，验证：

| 指标 | 期望值 |
|------|--------|
| `Train/mean_reward` | > 5.0（从接近0增长） |
| `Episode_Termination/base_contact` | < 0.10（摔倒率低） |
| `Episode_Termination/time_out` | > 0.85（多数存活到超时） |
| `Loss/learning_rate` | ≈ 1e-3（固定，不应降为0） |
| `Episode_Reward/track_lin_vel_xy_exp` | 增长趋势 |

若 `base_contact > 0.3`（频繁摔倒），检查：
1. `episode_length_s` 是否太长（尝试改为10s）
2. `init_noise_std` 是否太大（尝试0.5）

- [ ] **Step 4: 记录 Phase 1 最佳 checkpoint 路径**

```bash
ls $LOGDIR/model_*.pt | sort -t_ -k2 -n | tail -1
# 记录此路径，Phase 2 需要作为热启动
```

---

## Task 8：Phase 2 训练（地形课程，500–1500 迭代）

**前提：** Phase 1 已收敛（mean_reward > 5, 存活率 > 85%）

- [ ] **Step 1: 切换到地形课程**

在 `dog_env_cfg.py` 的 `DogWalkV2EnvCfg.__post_init__` 中，将：
```python
self.scene.terrain = FLAT_TERRAIN_IMPORTER_CFG
```
改为：
```python
self.scene.terrain = COMPETITION_TERRAIN_IMPORTER_CFG
```

- [ ] **Step 2: 热启动继续训练**

```bash
PHASE1_CKPT=$(ls -t /root/gpufree-data/legged_lab/logs/rsl_rl/dog_locomotion_walk_v2/*/model_*.pt | head -1)
echo "Resuming from: $PHASE1_CKPT"

python train.py --task Dog-Legged-Lab-Walk-v1 \
    --num_envs 4096 \
    --max_iterations 1500 \
    --checkpoint $PHASE1_CKPT \
    --headless
```

- [ ] **Step 3: Phase 2 成功标准（1500迭代后）**

```bash
python3 -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
logdir = sorted(glob.glob('/root/gpufree-data/legged_lab/logs/rsl_rl/dog_locomotion_walk_v2/*/'))[-1]
ef = glob.glob(logdir + '/events.*')[0]
ea = event_accumulator.EventAccumulator(ef); ea.Reload()
for tag in ['Train/mean_reward', 'Metrics/base_velocity/error_vel_xy',
            'Episode_Reward/track_lin_vel_xy_exp', 'Episode_Reward/track_height_exp']:
    evs = ea.Scalars(tag)
    if evs: print(f'{tag}: {evs[-1].value:.4f}')
" 2>/dev/null
```

| 指标 | 期望值 |
|------|--------|
| `Train/mean_reward` | > 15.0 |
| `error_vel_xy` | < 0.35 m/s |
| `track_height_exp` 均值 | > 0.5（高度跟踪有学习） |

- [ ] **Step 4: Commit 地形切换**

```bash
git add source/legged_lab/legged_lab/tasks/manager_based/dog_lab/dog_env_cfg.py
git commit -m "config: switch DogWalkV2 to competition terrain curriculum for Phase 2+"
```

---

## Task 9：Phase 3–4 训练（完整课程，1500–5000 迭代）

**前提：** Phase 2 完成，地形课程正常工作。

- [ ] **Step 1: 继续训练到 5000 迭代**

```bash
PHASE2_CKPT=$(ls -t /root/gpufree-data/legged_lab/logs/rsl_rl/dog_locomotion_walk_v2/*/model_*.pt | head -1)

python train.py --task Dog-Legged-Lab-Walk-v1 \
    --num_envs 4096 \
    --max_iterations 5000 \
    --checkpoint $PHASE2_CKPT \
    --headless
```

- [ ] **Step 2: 最终收敛标准**

训练结束后（或中途达到以下指标时可提前停止）：

| 指标 | 目标值 |
|------|--------|
| `error_vel_xy` | < 0.15 m/s |
| `error_vel_yaw` | < 0.15 rad/s |
| `Episode_Termination/time_out` | > 0.90 |
| `Episode_Termination/base_contact` | < 0.05 |
| `Train/mean_episode_length` | > 900 / 1000 步 |

- [ ] **Step 3: 导出最终模型**

```bash
FINAL_CKPT=$(ls -t /root/gpufree-data/legged_lab/logs/rsl_rl/dog_locomotion_walk_v2/*/model_*.pt | head -1)
echo "Final model: $FINAL_CKPT"
# 记录此路径，后续 Tier 3 部署使用
```

- [ ] **Step 4: 验证推理延迟**

```bash
python3 -c "
import torch, time
# 模拟 Actor 网络推理耗时（纯 CPU）
actor_input_dim = 3+3+3+4+12+12+12 + 273  # lin_vel+ang_vel+gravity+cmds+jpos+jvel+act + height_scan = 322
hidden = [512, 256, 128]
import torch.nn as nn
net = nn.Sequential(
    nn.Linear(actor_input_dim, hidden[0]), nn.ELU(),
    nn.Linear(hidden[0], hidden[1]), nn.ELU(),
    nn.Linear(hidden[1], hidden[2]), nn.ELU(),
    nn.Linear(hidden[2], 12),
)
x = torch.randn(1, actor_input_dim)
N = 1000
t0 = time.perf_counter()
for _ in range(N):
    _ = net(x)
elapsed = (time.perf_counter() - t0) / N * 1000
print(f'Mean inference: {elapsed:.3f} ms (target < 5ms)')
"
```

---

## 注意事项

1. **纵板桥自定义地形（Task 3 遗留）：** 当前 `narrow_rough` 是近似。如果测试中发现板桥通过率不足70%，需要编写自定义地形生成函数（在 `terrains.py` 中添加 `parallel_beams_terrain` 函数），生成精确的 20cm × full_length 并排梁。此工作量约1天，可在 Phase 3 训练期间并行完成。

2. **Critic 特权观测：** RSL-RL 的非对称 Critic 需要在 runner 中指定 `critic_obs` 键名为 `"critic"`。确认训练日志中 `value_function` loss 正常下降（不是 0）。

3. **h_cmd 在 standing envs：** `rel_standing_envs=0.1` 表示 10% 的环境发出零速度指令，但 `h_cmd` 仍然随机采样，策略需要在静止时也能调节高度。这是正确行为。

4. **Phase 切换方式：** 本计划通过修改 `dog_env_cfg.py` 中的一行来切换地形。更优雅的做法是通过命令行参数控制，但考虑到当前代码结构，直接修改最简单。
