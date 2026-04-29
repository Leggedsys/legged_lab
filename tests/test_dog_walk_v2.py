"""Smoke tests for Dog-Legged-Lab-Walk-v1 environment.

These tests verify that the environment is correctly registered and that
the key configuration classes are importable and have the expected values,
without starting Isaac Sim.
"""
import pytest


def test_env_registered():
    import gymnasium as gym
    all_ids = gym.registry.keys()
    assert "Dog-Legged-Lab-Walk-v1" in all_ids, (
        f"Dog-Legged-Lab-Walk-v1 not in registry. "
        f"Available Dog envs: {[k for k in all_ids if 'Dog' in k]}"
    )


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
    assert hasattr(cfg.scene, "height_scanner")
    assert hasattr(cfg.commands.base_velocity.ranges, "height")
    assert cfg.commands.base_velocity.ranges.height == (0.24, 0.32)


def test_reward_cfg_has_height_tracking():
    from legged_lab.tasks.manager_based.dog_lab.dog_env_cfg import DogWalkV2RewardsCfg
    cfg = DogWalkV2RewardsCfg()
    assert hasattr(cfg, "track_height_exp")
    assert cfg.track_height_exp.weight == pytest.approx(1.5)
    assert hasattr(cfg, "foot_slip")
    assert cfg.foot_slip.weight < 0
