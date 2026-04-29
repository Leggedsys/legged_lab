# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["enable_command_range"]


def enable_command_range(
    env: "ManagerBasedRLEnv", env_ids, old_value, target_value, num_steps: int
):
    """Switch a command range to the target value after the given number of steps."""
    from isaaclab.envs.mdp import modify_env_param

    if env.common_step_counter > num_steps:
        return target_value
    return modify_env_param.NO_CHANGE
