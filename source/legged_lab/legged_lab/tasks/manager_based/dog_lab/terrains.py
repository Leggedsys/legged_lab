"""Terrain configurations for Dog-Legged-Lab."""

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
    terrain_generator=TerrainGeneratorCfg(
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
    ),
    max_init_terrain_level=9,
    collision_group=-1,
    physics_material=_PHYSICS_MAT,
    debug_vis=False,
)
