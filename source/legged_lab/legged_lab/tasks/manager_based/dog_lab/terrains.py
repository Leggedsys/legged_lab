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
# Flat terrain (Phase 1)
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
# Full competition terrain curriculum (Phase 2-4)
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
    curriculum=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.1),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
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
        "stairs_up": HfPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.10),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_down": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.10),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_height_max=0.03,
            stone_width_range=(0.35, 0.50),
            stone_distance_range=(0.02, 0.10),
            border_width=0.25,
        ),
        "narrow_rough": HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.00, 0.02),
            noise_step=0.01,
            border_width=0.5,
        ),
    },
)

# ---------------------------------------------------------------------------
# Terrain Importer helpers
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
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=_PHYSICS_MAT,
    debug_vis=False,
)

# ---------------------------------------------------------------------------
# Transition terrain (Phase 1.5) — flat-dominant with mild roughness/slopes
# ---------------------------------------------------------------------------
TRANSITION_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.5),
        "mild_rough": HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.0, 0.03),
            noise_step=0.005,
            border_width=0.25,
        ),
        "gentle_slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.12),
            platform_width=2.0,
            border_width=0.25,
        ),
        "gentle_slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.12),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)

TRANSITION_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TRANSITION_TERRAIN_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=_PHYSICS_MAT,
    debug_vis=False,
)

# ---------------------------------------------------------------------------
# Phase 1.75 terrain — introduces stairs + stepping stones at low difficulty
# ---------------------------------------------------------------------------
PHASE1P75_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.0, 0.04),
            noise_step=0.005,
            border_width=0.25,
        ),
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.20),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.20),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": HfPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.06),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_down": HfInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.06),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stepping_stones": HfSteppingStonesTerrainCfg(
            proportion=0.10,
            stone_height_max=0.03,
            stone_width_range=(0.35, 0.50),
            stone_distance_range=(0.02, 0.10),
            border_width=0.25,
        ),
    },
)

PHASE1P75_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=PHASE1P75_TERRAIN_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=_PHYSICS_MAT,
    debug_vis=False,
)
