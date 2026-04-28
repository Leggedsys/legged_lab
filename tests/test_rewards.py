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
