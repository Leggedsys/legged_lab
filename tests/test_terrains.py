import pytest


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
