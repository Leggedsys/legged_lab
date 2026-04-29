# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

__all__ = ["joint_pos_target_l2", "track_height_exp", "foot_slip"]


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
