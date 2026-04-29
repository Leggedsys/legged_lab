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
    """command tensor shape should be 4 (vx, vy, ω, h)"""
    from legged_lab.tasks.manager_based.dog_lab.mdp.commands import (
        UniformVelocityHeightCommand,
        UniformVelocityHeightCommandCfg,
    )
    # Just check the class attribute exists (full test needs Isaac Sim)
    assert hasattr(UniformVelocityHeightCommand, "_build_command_buffer")
