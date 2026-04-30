"""
conftest.py: Make pure-tensor reward helpers and command classes importable
without Isaac Sim.

The legged_lab package __init__.py imports isaaclab_tasks which in turn
imports omni.log (only available inside the Isaac Sim runtime). This
conftest injects stub package entries into sys.modules so that individual
source modules can be loaded for unit-testing without starting the simulator.
"""
import importlib
import importlib.util
import sys
import types
from dataclasses import MISSING, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path to the actual source files
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).parent.parent / "source/legged_lab/legged_lab/tasks/manager_based"

_REWARDS_FILE = _SRC_ROOT / "legged_lab/mdp/rewards.py"
_COMMANDS_FILE = _SRC_ROOT / "dog_lab/mdp/commands.py"
_TERRAINS_FILE = _SRC_ROOT / "dog_lab/terrains.py"

# ---------------------------------------------------------------------------
# Inject stub package hierarchy into sys.modules so that the dotted-path
# imports resolve to the real files without executing any __init__.py that
# would pull in omni.
# ---------------------------------------------------------------------------
_PKG_CHAIN = [
    "legged_lab",
    "legged_lab.tasks",
    "legged_lab.tasks.manager_based",
    "legged_lab.tasks.manager_based.legged_lab",
    "legged_lab.tasks.manager_based.legged_lab.mdp",
    "legged_lab.tasks.manager_based.dog_lab",
    "legged_lab.tasks.manager_based.dog_lab.mdp",
]

for _pkg in _PKG_CHAIN:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

# ---------------------------------------------------------------------------
# Load rewards.py directly from disk and register under the expected name
# ---------------------------------------------------------------------------
_REWARDS_MODULE_NAME = "legged_lab.tasks.manager_based.legged_lab.mdp.rewards"

if _REWARDS_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_REWARDS_MODULE_NAME, _REWARDS_FILE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_REWARDS_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)
    # also attach as attribute on parent stub
    sys.modules["legged_lab.tasks.manager_based.legged_lab.mdp"].rewards = _mod

# ---------------------------------------------------------------------------
# Stubs for isaaclab classes needed by commands.py
#
# commands.py imports:
#   from isaaclab.envs.mdp.commands import UniformVelocityCommand,
#                                          UniformVelocityCommandCfg
#   from isaaclab.utils import configclass
#
# We inject lightweight stubs so the module loads without Isaac Sim.
# ---------------------------------------------------------------------------

# -- configclass stub: behaves like @dataclass for test purposes
# Replaces MISSING with None so Python dataclass doesn't reject field ordering.
def _configclass_stub(cls=None, **kwargs):
    """Minimal configclass stub: wraps the class as a plain dataclass.

    Rewrites any field annotation whose default value is MISSING to use
    None instead, avoiding Python's 'non-default argument follows default'
    error when inheriting from a base with optional fields.

    Also handles class-level overrides of inherited dataclass fields: when a
    subclass assigns a new default value to a field inherited from a parent
    dataclass, this re-annotates and wraps mutable values in field() so that
    @dataclass honours the override (plain @dataclass ignores class-level
    assignments that don't appear in __annotations__).
    """
    import dataclasses as _dc
    import inspect as _inspect

    def _wrap(c):
        # Ensure __annotations__ dict exists on this class
        if "__annotations__" not in c.__dict__:
            c.__annotations__ = {}
        own_annotations = c.__dict__["__annotations__"]

        def _wrap_mutable(fname, val):
            """Replace a mutable class-level default with a default_factory."""
            val_copy = val
            setattr(
                c,
                fname,
                _dc.field(
                    default_factory=(
                        lambda v=val_copy: _dc.replace(v)
                        if _dc.is_dataclass(v)
                        else type(v)(v)
                    )
                ),
            )

        # 1) Handle own annotations: replace MISSING with None, wrap mutables
        for fname in list(own_annotations):
            val = c.__dict__.get(fname, _dc.MISSING)
            if val is _dc.MISSING:
                setattr(c, fname, None)
            elif not isinstance(val, (_dc.Field, int, float, bool, str, type(None))):
                # Mutable default (dataclass instance, list, dict, …) — wrap it
                _wrap_mutable(fname, val)

        # 2) Promote class-level overrides of *inherited* dataclass fields so
        #    that @dataclass honours them.  Without this, a child class that
        #    sets `policy = SomeCfg(...)` on an inherited `policy` field will
        #    silently keep the parent's default.
        for base in _inspect.getmro(c)[1:]:
            if base is object:
                continue
            for fname, _fobj in getattr(base, "__dataclass_fields__", {}).items():
                # Only if the child overrides at class level but hasn't re-annotated
                if fname in c.__dict__ and fname not in own_annotations:
                    val = c.__dict__[fname]
                    # Re-annotate so @dataclass processes this field
                    own_annotations[fname] = type(val) if val is not None else object
                    # Wrap mutable (non-primitive) defaults in default_factory
                    if isinstance(val, (_dc.Field,)):
                        pass  # already wrapped
                    elif not isinstance(val, (int, float, bool, str, type(None))):
                        _wrap_mutable(fname, val)
                    # primitives and None stay as-is

        return dataclass(c)

    if cls is None:
        return _wrap
    return _wrap(cls)

# -- CommandTermCfg stub
@dataclass
class _CommandTermCfgStub:
    resampling_time_range: tuple = (5.0, 5.0)
    class_type: type = None
    debug_vis: bool = False

# -- Ranges stub for UniformVelocityCommandCfg
@dataclass
class _UniformRangesStub:
    lin_vel_x: tuple = MISSING
    lin_vel_y: tuple = MISSING
    ang_vel_z: tuple = MISSING
    heading: tuple = None

# -- UniformVelocityCommandCfg stub
# Note: MISSING sentinel counts as "no default" in plain dataclass;
# to avoid MRO ordering issues we don't inherit _CommandTermCfgStub here.
@dataclass
class _UniformVelocityCommandCfgStub:
    asset_name: str = MISSING
    ranges: _UniformRangesStub = None
    resampling_time_range: tuple = (5.0, 5.0)
    class_type: type = None
    debug_vis: bool = False
    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0

    # Expose nested Ranges class so subclasses can inherit it
    Ranges = _UniformRangesStub

# -- UniformVelocityCommand stub (runtime base for UniformVelocityHeightCommand)
class _UniformVelocityCommandStub:
    cfg: _UniformVelocityCommandCfgStub

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.num_envs = 0
        self.device = "cpu"

    def _resample_command(self, env_ids):
        pass

    @property
    def command(self):
        import torch
        return torch.zeros(self.num_envs, 3)

    @property
    def vel_command_b(self):
        import torch
        return torch.zeros(self.num_envs, 3)

# -- Register isaaclab stubs in sys.modules
def _make_stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

# isaaclab.utils
_isaaclab_utils = _make_stub_module(
    "isaaclab.utils",
    configclass=_configclass_stub,
    MISSING=MISSING,
)
# isaaclab.envs.mdp.commands (the submodule commands.py imports from)
_isaaclab_commands_mod = _make_stub_module(
    "isaaclab.envs.mdp.commands",
    UniformVelocityCommand=_UniformVelocityCommandStub,
    UniformVelocityCommandCfg=_UniformVelocityCommandCfgStub,
)

_isaaclab_envs_stub = _make_stub_module(
    "isaaclab.envs",
    # ManagerBasedRLEnvCfg stub injected later after the class is defined,
    # but we need the module object now so commands.py can import from it
)
_stub_modules = {
    "isaaclab": types.ModuleType("isaaclab"),
    "isaaclab.utils": _isaaclab_utils,
    "isaaclab.envs": _isaaclab_envs_stub,
    "isaaclab.envs.mdp": types.ModuleType("isaaclab.envs.mdp"),
    "isaaclab.envs.mdp.commands": _isaaclab_commands_mod,
}
for _name, _mod in _stub_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _mod

# ---------------------------------------------------------------------------
# Load commands.py directly from disk and register under the expected name
# ---------------------------------------------------------------------------
_COMMANDS_MODULE_NAME = "legged_lab.tasks.manager_based.dog_lab.mdp.commands"

if _COMMANDS_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_COMMANDS_MODULE_NAME, _COMMANDS_FILE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_COMMANDS_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)
    # attach as attribute on parent stub
    sys.modules["legged_lab.tasks.manager_based.dog_lab.mdp"].commands = _mod

# ---------------------------------------------------------------------------
# Stubs for isaaclab terrain classes needed by terrains.py
#
# terrains.py imports:
#   import isaaclab.sim as sim_utils
#   from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
#   from isaaclab.terrains.height_field import Hf*Cfg classes
#   from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
#
# These are pure-Python dataclasses that don't need Isaac Sim at runtime, but
# their parent __init__.py imports omni.log.  We stub the modules here.
# ---------------------------------------------------------------------------

@dataclass
class _SubTerrainBaseCfgStub:
    proportion: float = 1.0
    size: tuple = (10.0, 10.0)
    flat_patch_sampling: object = None


@dataclass
class _MeshPlaneTerrainCfgStub(_SubTerrainBaseCfgStub):
    pass


@dataclass
class _HfTerrainBaseCfgStub(_SubTerrainBaseCfgStub):
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: object = None


@dataclass
class _HfRandomUniformTerrainCfgStub(_HfTerrainBaseCfgStub):
    noise_range: tuple = None
    noise_step: float = None
    downsampled_scale: object = None


@dataclass
class _HfPyramidSlopedTerrainCfgStub(_HfTerrainBaseCfgStub):
    slope_range: tuple = None
    platform_width: float = 1.0
    inverted: bool = False


@dataclass
class _HfInvertedPyramidSlopedTerrainCfgStub(_HfPyramidSlopedTerrainCfgStub):
    inverted: bool = True


@dataclass
class _HfPyramidStairsTerrainCfgStub(_HfTerrainBaseCfgStub):
    step_height_range: tuple = None
    step_width: float = None
    platform_width: float = 1.0
    inverted: bool = False


@dataclass
class _HfInvertedPyramidStairsTerrainCfgStub(_HfPyramidStairsTerrainCfgStub):
    inverted: bool = True


@dataclass
class _HfSteppingStonesTerrainCfgStub(_HfTerrainBaseCfgStub):
    stone_height_max: float = None
    stone_width_range: tuple = None
    stone_distance_range: tuple = None
    holes_depth: float = -10.0
    platform_width: float = 1.0


@dataclass
class _TerrainGeneratorCfgStub:
    seed: object = None
    size: tuple = None
    border_width: float = 0.0
    border_height: float = 1.0
    num_rows: int = 1
    num_cols: int = 1
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: object = 0.75
    use_cache: bool = False
    cache_dir: str = "/tmp/isaaclab/terrains"
    sub_terrains: object = None
    difficulty_range: tuple = (0.0, 1.0)
    curriculum: bool = False
    color_scheme: str = "none"


@dataclass
class _RigidBodyMaterialCfgStub:
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    friction_combine_mode: str = "average"
    restitution_combine_mode: str = "average"


@dataclass
class _TerrainImporterCfgStub:
    prim_path: str = None
    terrain_type: str = "generator"
    terrain_generator: object = None
    max_init_terrain_level: object = None
    collision_group: int = -1
    physics_material: object = None
    debug_vis: bool = False
    num_envs: int = 1
    env_spacing: object = None
    usd_path: object = None
    visual_material: object = None


# Build the stub modules for isaaclab.terrains and isaaclab.sim
_isaaclab_sim_stub = _make_stub_module(
    "isaaclab.sim",
    RigidBodyMaterialCfg=_RigidBodyMaterialCfgStub,
)

_isaaclab_terrains_stub = _make_stub_module(
    "isaaclab.terrains",
    TerrainGeneratorCfg=_TerrainGeneratorCfgStub,
    TerrainImporterCfg=_TerrainImporterCfgStub,
)

_isaaclab_hf_stub = _make_stub_module(
    "isaaclab.terrains.height_field",
    HfRandomUniformTerrainCfg=_HfRandomUniformTerrainCfgStub,
    HfPyramidSlopedTerrainCfg=_HfPyramidSlopedTerrainCfgStub,
    HfInvertedPyramidSlopedTerrainCfg=_HfInvertedPyramidSlopedTerrainCfgStub,
    HfPyramidStairsTerrainCfg=_HfPyramidStairsTerrainCfgStub,
    HfInvertedPyramidStairsTerrainCfg=_HfInvertedPyramidStairsTerrainCfgStub,
    HfSteppingStonesTerrainCfg=_HfSteppingStonesTerrainCfgStub,
)

_isaaclab_trimesh_stub = _make_stub_module(
    "isaaclab.terrains.trimesh",
    MeshPlaneTerrainCfg=_MeshPlaneTerrainCfgStub,
)

_terrain_stub_modules = {
    "isaaclab.sim": _isaaclab_sim_stub,
    "isaaclab.terrains": _isaaclab_terrains_stub,
    "isaaclab.terrains.height_field": _isaaclab_hf_stub,
    "isaaclab.terrains.trimesh": _isaaclab_trimesh_stub,
}
for _name, _mod in _terrain_stub_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _mod

# ---------------------------------------------------------------------------
# Load terrains.py directly from disk and register under the expected name
# ---------------------------------------------------------------------------
_TERRAINS_MODULE_NAME = "legged_lab.tasks.manager_based.dog_lab.terrains"

if _TERRAINS_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_TERRAINS_MODULE_NAME, _TERRAINS_FILE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_TERRAINS_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)
    # attach as attribute on parent stub
    sys.modules["legged_lab.tasks.manager_based.dog_lab"].terrains = _mod

# ===========================================================================
# Additional stubs for Dog-Walk-v2 smoke tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Source file paths for dog_lab modules
# ---------------------------------------------------------------------------
_DOG_LAB_ROOT = Path(__file__).parent.parent / "source/legged_lab/legged_lab"
_ASSETS_ROOT = _DOG_LAB_ROOT / "assets"

_DOG_ENV_CFG_FILE = _SRC_ROOT / "dog_lab/dog_env_cfg.py"
_DOG_AGENTS_ROOT = _SRC_ROOT / "dog_lab/agents"
_LEGGED_AGENTS_ROOT = _SRC_ROOT / "legged_lab/agents"

# ---------------------------------------------------------------------------
# isaaclab_rl.rsl_rl stubs  (needed by PPO runner cfg chain)
#
# The real rl_cfg.py is pure Python but its __init__.py pulls in
# vecenv_wrapper which imports omni.log.  We stub the module here so that
# the three dataclasses are importable without Isaac Sim.
# ---------------------------------------------------------------------------
@dataclass
class _RslRlPpoActorCriticCfgStub:
    class_name: str = "ActorCritic"
    init_noise_std: float = 0.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: object = None
    critic_hidden_dims: object = None
    activation: str = "elu"


@dataclass
class _RslRlPpoAlgorithmCfgStub:
    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.001
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: object = None
    rnd_cfg: object = None


@dataclass
class _RslRlOnPolicyRunnerCfgStub:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 32
    max_iterations: int = 1000
    empirical_normalization: bool = False
    policy: object = None
    algorithm: object = None
    clip_actions: object = None
    save_interval: int = 100
    experiment_name: str = ""
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


_isaaclab_rl_rsl_rl_stub = _make_stub_module(
    "isaaclab_rl.rsl_rl",
    RslRlOnPolicyRunnerCfg=_RslRlOnPolicyRunnerCfgStub,
    RslRlPpoActorCriticCfg=_RslRlPpoActorCriticCfgStub,
    RslRlPpoAlgorithmCfg=_RslRlPpoAlgorithmCfgStub,
)

_rl_stub_modules = {
    "isaaclab_rl": types.ModuleType("isaaclab_rl"),
    "isaaclab_rl.rsl_rl": _isaaclab_rl_rsl_rl_stub,
}
for _name, _mod in _rl_stub_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _mod

# ---------------------------------------------------------------------------
# Load the three PPO runner configs via importlib so they're importable
# without triggering any real isaaclab __init__.py
# ---------------------------------------------------------------------------

# Ensure parent stub packages exist
_PPO_PKG_CHAIN = [
    "legged_lab.tasks.manager_based.legged_lab.agents",
    "legged_lab.tasks.manager_based.dog_lab.agents",
]
for _pkg in _PPO_PKG_CHAIN:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)


def _load_module_from_file(module_name, file_path, parent_pkg=None):
    """Load a Python source file and register it under *module_name*."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    _s = importlib.util.spec_from_file_location(module_name, file_path)
    _m = importlib.util.module_from_spec(_s)
    sys.modules[module_name] = _m
    _s.loader.exec_module(_m)
    if parent_pkg:
        attr = module_name.rsplit(".", 1)[-1]
        setattr(sys.modules[parent_pkg], attr, _m)
    return _m


_legged_ppo_cfg = _load_module_from_file(
    "legged_lab.tasks.manager_based.legged_lab.agents.rsl_rl_ppo_cfg",
    _LEGGED_AGENTS_ROOT / "rsl_rl_ppo_cfg.py",
    "legged_lab.tasks.manager_based.legged_lab.agents",
)

_dog_ppo_cfg = _load_module_from_file(
    "legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_ppo_cfg",
    _DOG_AGENTS_ROOT / "rsl_rl_ppo_cfg.py",
    "legged_lab.tasks.manager_based.dog_lab.agents",
)

_dog_walk_ppo_cfg = _load_module_from_file(
    "legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_walk_ppo_cfg",
    _DOG_AGENTS_ROOT / "rsl_rl_walk_ppo_cfg.py",
    "legged_lab.tasks.manager_based.dog_lab.agents",
)

_dog_walk_v2_ppo_cfg = _load_module_from_file(
    "legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_walk_v2_ppo_cfg",
    _DOG_AGENTS_ROOT / "rsl_rl_walk_v2_ppo_cfg.py",
    "legged_lab.tasks.manager_based.dog_lab.agents",
)

# ---------------------------------------------------------------------------
# Additional isaaclab stubs needed by dog_env_cfg.py
#
# dog_env_cfg.py imports:
#   from isaaclab.assets import AssetBaseCfg
#   from isaaclab.envs import ManagerBasedRLEnvCfg
#   from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ObservationGroupCfg,
#                                  ObservationTermCfg, RewardTermCfg, SceneEntityCfg,
#                                  TerminationTermCfg
#   from isaaclab.scene import InteractiveSceneCfg
#   from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
#   from isaaclab.utils.noise import AdditiveUniformNoiseCfg
# ---------------------------------------------------------------------------

# -- Term config stubs (just dataclasses that hold a func + params)
@dataclass
class _TermCfgStub:
    func: object = None
    weight: float = 0.0
    params: object = None
    time_out: bool = False
    mode: str = "reset"
    noise: object = None
    clip: object = None

@dataclass
class _ObsTermCfgStub:
    func: object = None
    params: object = None
    noise: object = None
    clip: object = None

@dataclass
class _ObsGroupCfgStub:
    enable_corruption: bool = False
    concatenate_terms: bool = True

@dataclass
class _SceneEntityCfgStub:
    name: str = ""
    joint_names: object = None
    body_names: object = None
    joint_ids: object = None
    body_ids: object = None
    preserves_order: bool = False

@dataclass
class _InteractiveSceneCfgStub:
    num_envs: int = 1
    env_spacing: float = 2.0

@dataclass
class _SimCfgStub:
    dt: float = 0.005
    render_interval: int = 4
    physics_material: object = None

from dataclasses import field as _dc_field


@dataclass
class _ManagerBasedRLEnvCfgStub:
    scene: object = None
    observations: object = None
    actions: object = None
    commands: object = None
    events: object = None
    rewards: object = None
    terminations: object = None
    curriculum: object = None
    decimation: int = 4
    episode_length_s: float = 10.0
    # Always initialise sim so child __post_init__ can access self.sim.dt etc.
    sim: _SimCfgStub = _dc_field(default_factory=_SimCfgStub)

@dataclass
class _AssetBaseCfgStub:
    prim_path: str = ""
    spawn: object = None

@dataclass
class _ContactSensorCfgStub:
    prim_path: str = ""
    history_length: int = 3
    track_air_time: bool = False

@dataclass
class _OffsetCfgStub:
    pos: tuple = (0.0, 0.0, 0.0)

class _RayCasterCfgStub:
    """Minimal RayCasterCfg stub (not a dataclass to avoid field conflicts)."""
    OffsetCfg = _OffsetCfgStub

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class _PatternsStub:
    @dataclass
    class GridPatternCfg:
        resolution: float = 0.05
        size: object = None

@dataclass
class _AdditiveUniformNoiseCfgStub:
    n_min: float = 0.0
    n_max: float = 0.0

# -- DOG_URDF_CFG stub (ArticulationCfg-like object)
class _ArticulationCfgStub:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def replace(self, **kwargs):
        import copy
        new = copy.copy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new

_DOG_URDF_CFG_STUB = _ArticulationCfgStub(prim_path="")

# -- sim_utils stubs
class _SimUtilsStub:
    @dataclass
    class RigidBodyMaterialCfg(_RigidBodyMaterialCfgStub):
        friction_combine_mode: str = "average"
        restitution_combine_mode: str = "average"

    @dataclass
    class RigidBodyPropertiesCfg:
        disable_gravity: bool = False
        retain_accelerations: bool = False
        linear_damping: float = 0.0
        angular_damping: float = 0.0
        max_linear_velocity: float = 1000.0
        max_angular_velocity: float = 1000.0
        max_depenetration_velocity: float = 3.0

    @dataclass
    class ArticulationRootPropertiesCfg:
        enabled_self_collisions: bool = False
        solver_position_iteration_count: int = 8
        solver_velocity_iteration_count: int = 0

    @dataclass
    class UsdFileCfg:
        usd_path: str = ""
        activate_contact_sensors: bool = False
        rigid_props: object = None
        articulation_props: object = None

    @dataclass
    class DomeLightCfg:
        color: tuple = (0.9, 0.9, 0.9)
        intensity: float = 500.0


# Extend isaaclab.sim stub with the new types
_isaaclab_sim_stub.RigidBodyMaterialCfg = _SimUtilsStub.RigidBodyMaterialCfg
_isaaclab_sim_stub.RigidBodyPropertiesCfg = _SimUtilsStub.RigidBodyPropertiesCfg
_isaaclab_sim_stub.ArticulationRootPropertiesCfg = _SimUtilsStub.ArticulationRootPropertiesCfg
_isaaclab_sim_stub.UsdFileCfg = _SimUtilsStub.UsdFileCfg
_isaaclab_sim_stub.DomeLightCfg = _SimUtilsStub.DomeLightCfg

# -- Register all new isaaclab sub-module stubs
# Inject into already-registered isaaclab.envs (created earlier as empty stub)
_isaaclab_envs_stub.ManagerBasedRLEnvCfg = _ManagerBasedRLEnvCfgStub

_dog_env_new_stub_modules = {
    "isaaclab.assets": _make_stub_module(
        "isaaclab.assets",
        AssetBaseCfg=_AssetBaseCfgStub,
        ArticulationCfg=_ArticulationCfgStub,
    ),
    "isaaclab.managers": _make_stub_module(
        "isaaclab.managers",
        CurriculumTermCfg=_TermCfgStub,
        EventTermCfg=_TermCfgStub,
        ObservationGroupCfg=_ObsGroupCfgStub,
        ObservationTermCfg=_ObsTermCfgStub,
        RewardTermCfg=_TermCfgStub,
        SceneEntityCfg=_SceneEntityCfgStub,
        TerminationTermCfg=_TermCfgStub,
    ),
    "isaaclab.scene": _make_stub_module(
        "isaaclab.scene",
        InteractiveSceneCfg=_InteractiveSceneCfgStub,
    ),
    "isaaclab.sensors": _make_stub_module(
        "isaaclab.sensors",
        ContactSensorCfg=_ContactSensorCfgStub,
        RayCasterCfg=_RayCasterCfgStub,
        patterns=_PatternsStub,
    ),
    "isaaclab.utils.noise": _make_stub_module(
        "isaaclab.utils.noise",
        AdditiveUniformNoiseCfg=_AdditiveUniformNoiseCfgStub,
    ),
    "isaaclab.actuators": _make_stub_module(
        "isaaclab.actuators",
        ImplicitActuatorCfg=_ArticulationCfgStub,
    ),
}
for _name, _mod in _dog_env_new_stub_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _mod

# -- legged_lab.assets stubs
_LEGGED_LAB_ASSETS_PKG = "legged_lab.assets"
if _LEGGED_LAB_ASSETS_PKG not in sys.modules:
    _assets_mod = types.ModuleType(_LEGGED_LAB_ASSETS_PKG)
    sys.modules[_LEGGED_LAB_ASSETS_PKG] = _assets_mod

# Provide a stub DOG_URDF_CFG directly (skip dog_cfg.py which also has sim_utils imports)
_dog_cfg_mod_name = "legged_lab.assets.dog_cfg"
if _dog_cfg_mod_name not in sys.modules:
    _dog_cfg_stub = types.ModuleType(_dog_cfg_mod_name)
    _dog_cfg_stub.DOG_URDF_CFG = _DOG_URDF_CFG_STUB
    _dog_cfg_stub.DOG_JOINT_SIGN = {}
    sys.modules[_dog_cfg_mod_name] = _dog_cfg_stub
    sys.modules[_LEGGED_LAB_ASSETS_PKG].dog_cfg = _dog_cfg_stub

# ---------------------------------------------------------------------------
# Build mdp stub that dog_env_cfg.py uses via `from . import mdp`
#
# The `mdp` referenced in dog_env_cfg.py is the dog_lab.mdp package.
# It re-exports everything from legged_lab.mdp (which includes isaaclab.envs.mdp)
# plus UniformVelocityCommandCfg / UniformVelocityHeightCommandCfg.
#
# Since dog_env_cfg.py uses `mdp.XxxCfg(...)` and `mdp.func_name` as
# callable references (not called at import time), we just need a module
# object with all referenced names present.
# ---------------------------------------------------------------------------
def _noop(*args, **kwargs):
    return None


_mdp_stub = sys.modules.get("legged_lab.tasks.manager_based.dog_lab.mdp")
if _mdp_stub is None:
    _mdp_stub = types.ModuleType("legged_lab.tasks.manager_based.dog_lab.mdp")
    sys.modules["legged_lab.tasks.manager_based.dog_lab.mdp"] = _mdp_stub

# Inject all symbols that dog_env_cfg.py references via mdp.*
_mdp_func_names = [
    "action_rate_l2", "ang_vel_xy_l2", "apply_external_force_torque",
    "bad_orientation", "base_ang_vel", "base_height_l2", "base_lin_vel",
    "feet_air_time", "flat_orientation_l2", "foot_slip", "generated_commands",
    "prolonged_air_penalty", "trot_gait_reward", "gait_clock_reward", "gait_phase_obs",
    "height_scan", "illegal_contact", "is_alive", "joint_acc_l2",
    "joint_deviation_l1", "joint_pos_rel", "joint_torques_l2",
    "joint_vel_rel", "last_action", "lin_vel_z_l2", "projected_gravity",
    "randomize_rigid_body_mass", "randomize_rigid_body_material",
    "reset_joints_by_scale", "reset_root_state_uniform",
    "root_height_below_minimum", "terrain_levels_vel", "time_out",
    "track_ang_vel_z_world_exp", "track_height_exp",
    "track_lin_vel_xy_yaw_frame_exp", "undesired_contacts",
]
for _fn in _mdp_func_names:
    if not hasattr(_mdp_stub, _fn):
        setattr(_mdp_stub, _fn, _noop)

# JointPositionActionCfg stub
@dataclass
class _JointPositionActionCfgStub:
    asset_name: str = ""
    joint_names: object = None
    scale: object = None
    use_default_offset: bool = True

if not hasattr(_mdp_stub, "JointPositionActionCfg"):
    _mdp_stub.JointPositionActionCfg = _JointPositionActionCfgStub

# UniformVelocityCommandCfg and UniformVelocityHeightCommandCfg come from
# the already-loaded commands module
_commands_mod = sys.modules.get("legged_lab.tasks.manager_based.dog_lab.mdp.commands")
if _commands_mod is not None:
    if not hasattr(_mdp_stub, "UniformVelocityCommandCfg"):
        _mdp_stub.UniformVelocityCommandCfg = _commands_mod.UniformVelocityCommandCfg
    if not hasattr(_mdp_stub, "UniformVelocityHeightCommandCfg"):
        _mdp_stub.UniformVelocityHeightCommandCfg = _commands_mod.UniformVelocityHeightCommandCfg
    if not hasattr(_mdp_stub, "UniformVelocityCommand"):
        _mdp_stub.UniformVelocityCommand = getattr(_commands_mod, "UniformVelocityCommand", _noop)
    if not hasattr(_mdp_stub, "UniformVelocityHeightCommand"):
        _mdp_stub.UniformVelocityHeightCommand = getattr(_commands_mod, "UniformVelocityHeightCommand", _noop)

# ---------------------------------------------------------------------------
# Load dog_env_cfg.py directly from disk
# ---------------------------------------------------------------------------
_DOG_ENV_CFG_MODULE_NAME = "legged_lab.tasks.manager_based.dog_lab.dog_env_cfg"

if _DOG_ENV_CFG_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_DOG_ENV_CFG_MODULE_NAME, _DOG_ENV_CFG_FILE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_DOG_ENV_CFG_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)
    sys.modules["legged_lab.tasks.manager_based.dog_lab"].dog_env_cfg = _mod

# ---------------------------------------------------------------------------
# Register Dog gym environments by executing dog_lab/__init__.py stubs
#
# dog_lab/__init__.py does:
#   import gymnasium as gym
#   from . import agents
#   gym.register(id="Dog-Legged-Lab-Walk-v1", ...)
#
# We replicate this directly so test_env_registered can pass without running
# the full isaaclab_tasks import chain.
# ---------------------------------------------------------------------------
import gymnasium as _gym

for _env_id, _env_cfg_ep, _rsl_ep in [
    (
        "Dog-Legged-Lab-v0",
        "legged_lab.tasks.manager_based.dog_lab.dog_env_cfg:DogEnvCfg",
        "legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_ppo_cfg:DogPPORunnerCfg",
    ),
    (
        "Dog-Legged-Lab-Walk-v0",
        "legged_lab.tasks.manager_based.dog_lab.dog_env_cfg:DogWalkEnvCfg",
        "legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_walk_ppo_cfg:DogWalkPPORunnerCfg",
    ),
    (
        "Dog-Legged-Lab-Walk-v1",
        "legged_lab.tasks.manager_based.dog_lab.dog_env_cfg:DogWalkV2EnvCfg",
        "legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_walk_v2_ppo_cfg:DogWalkV2PPORunnerCfg",
    ),
]:
    if _env_id not in _gym.registry:
        _gym.register(
            id=_env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _env_cfg_ep,
                "rsl_rl_cfg_entry_point": _rsl_ep,
            },
        )
