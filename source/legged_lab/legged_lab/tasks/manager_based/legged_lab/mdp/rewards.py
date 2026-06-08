# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

__all__ = [
    "joint_pos_target_l2",
    "track_height_exp",
    "foot_slip",
    "prolonged_air_penalty",
    "trot_gait_reward",
    "gait_clock_reward",
    "gait_phase_obs",
    "flat_orientation_exp",
]


def joint_pos_target_l2(
    env: "ManagerBasedRLEnv", target: float, asset_cfg: "SceneEntityCfg"
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import wrap_to_pi

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


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
    """Exponential reward for tracking the height command (4th dim of velocity command).

    Requires the command tensor to have at least 4 columns; column 3 is the height
    target (h_cmd from UniformVelocityHeightCommand).
    """
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.assets import Articulation

    if asset_cfg is None:
        asset_cfg = _SceneEntityCfg("robot")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (N, >=4)
    assert command.shape[1] >= 4, (
        f"track_height_exp requires command with >=4 dims, got shape {command.shape}"
    )
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


def prolonged_air_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg" = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize each foot that has been continuously airborne beyond threshold seconds.

    Encourages all feet to periodically make ground contact, breaking 3-legged gaits.
    The penalty grows linearly with excess air time, so a permanently suspended leg
    accumulates a large per-step cost without constraining normal swing amplitude.
    """
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.sensors import ContactSensor

    if sensor_cfg is None:
        sensor_cfg = _SceneEntityCfg("contact_forces", body_names=".*_foot")

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # current_air_time: (N, num_bodies) — seconds each body has been continuously in air
    current_air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = (current_air_time - threshold).clamp(min=0.0)  # (N, F)
    return excess.sum(dim=1)


def gait_phase_obs(env: "ManagerBasedRLEnv", frequency: float = 1.5) -> torch.Tensor:
    """Return [sin(φ), cos(φ)] of the trot gait phase as a 2-dim observation.

    The policy uses this to synchronise its leg commands with the reference clock.
    φ = 2π * frequency * t  where t is the current step count × step_dt.
    """
    t = env.episode_length_buf * env.step_dt  # (N,)
    phase = 2.0 * math.pi * frequency * t
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # (N, 2)


def gait_clock_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg" = None,
    frequency: float = 1.5,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward foot contact pattern that matches a trot clock at the target frequency.

    Trot schedule (body order FL=0, FR=1, RL=2, RR=3):
      • When sin(φ) > 0 → FL and RR should be in contact, FR and RL in swing.
      • When sin(φ) < 0 → FR and RL should be in contact, FL and RR in swing.

    Returns the fraction of legs (0–1) whose contact state matches the schedule,
    scaled to zero when the robot is stationary.
    """
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.sensors import ContactSensor

    if sensor_cfg is None:
        sensor_cfg = _SceneEntityCfg("contact_forces", body_names=".*_foot")

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 4, 3)
    in_contact = forces.norm(dim=-1) > 1.0  # (N, 4)  bool

    t = env.episode_length_buf * env.step_dt
    phase = 2.0 * math.pi * frequency * t  # (N,)
    sin_p = torch.sin(phase)  # (N,)

    # Desired contact: 1=stance 0=swing
    fl_des = (sin_p > 0).float()
    rr_des = fl_des
    fr_des = (sin_p < 0).float()
    rl_des = fr_des
    desired = torch.stack([fl_des, fr_des, rl_des, rr_des], dim=1)  # (N, 4)

    match = (in_contact == desired.bool()).float().mean(dim=1)  # (N,)

    vel_cmd = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (vel_cmd.norm(dim=-1) > velocity_threshold).float()
    return moving * match


def trot_gait_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: "SceneEntityCfg" = None,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward diagonal foot contact synchronization to encourage trot gait.

    Trot: FL+RR in contact together, FR+RL in contact together.
    Body order assumed alphabetical from USD: FL(0), FR(1), RL(2), RR(3).
    Only active when the robot is commanded to move (above velocity_threshold).
    """
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.sensors import ContactSensor

    if sensor_cfg is None:
        sensor_cfg = _SceneEntityCfg("contact_forces", body_names=".*_foot")

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 4, 3)
    in_contact = forces.norm(dim=-1) > 1.0  # (N, 4)

    # Diagonal pairs: FL(0)+RR(3) and FR(1)+RL(2)
    fl, fr, rl, rr = in_contact[:, 0], in_contact[:, 1], in_contact[:, 2], in_contact[:, 3]
    diag1 = (fl == rr).float()
    diag2 = (fr == rl).float()

    vel_cmd = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (vel_cmd.norm(dim=-1) > velocity_threshold).float()
    return moving * (diag1 + diag2) * 0.5


def _flat_orientation_exp_impl(g_xy_sq: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Pure-tensor helper: :math:`\exp(-\|g_{xy}\|^2 / \sigma)`.

    Maps squared x-y gravity component to (0, 1] — 1 when perfectly level,
    decaying exponentially as the base tilts.
    """
    return torch.exp(-g_xy_sq / sigma)


def flat_orientation_exp(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.05,
    asset_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    r"""Exponential reward for keeping the base level.

    Uses the projected gravity vector (sim-to-real stable, no Euler-angle singularities).
    Returns :math:`\exp(-\|g_{xy}\|^2 / \sigma)` in (0, 1].

    Apply with a positive weight — the reward approaches 1 when level,
    decays to near 0 when severely tilted.  This gives a gentle gradient
    near the target while still strongly penalising large tilts.

    By default σ = 0.05, which means:
      * 5° tilt  → reward ≈ 0.99
      * 10° tilt → reward ≈ 0.94
      * 20° tilt → reward ≈ 0.78
      * 45° tilt → reward ≈ 0.37
    """
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    from isaaclab.assets import Articulation

    if asset_cfg is None:
        asset_cfg = _SceneEntityCfg("robot")
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b  # (N, 3)
    g_xy_sq = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)  # (N,)
    return _flat_orientation_exp_impl(g_xy_sq, sigma)
