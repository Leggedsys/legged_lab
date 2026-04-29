from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityHeightCommand(UniformVelocityCommand):
    """Velocity command extended with a height component.

    Outputs command tensor of shape (N, 4): [vx, vy, omega, h_cmd].
    The height is sampled uniformly from cfg.ranges.height.

    Note: standing environments (10% by default) have their velocity command zeroed by the
    parent class on each update step, but h_cmd retains its sampled value — this is
    intentional, as the policy should learn to regulate height even when stationary.
    """

    cfg: UniformVelocityHeightCommandCfg

    def __init__(self, cfg: UniformVelocityHeightCommandCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.h_command = torch.zeros(self.num_envs, device=self.device)
        # Initialize to nominal height (midpoint of range)
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
        """Shape (N, 4): [vx, vy, omega, h_cmd]."""
        return torch.cat([self.vel_command_b, self.h_command.unsqueeze(1)], dim=1)


@configclass
class UniformVelocityHeightCommandCfg(UniformVelocityCommandCfg):
    """Extends velocity command with a uniform height command (4th dimension)."""

    class_type: type = UniformVelocityHeightCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        height: tuple[float, float] = MISSING
        """Min and max target base height (m). e.g. (0.15, 0.30)"""
