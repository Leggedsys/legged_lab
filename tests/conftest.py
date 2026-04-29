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
    """
    import dataclasses as _dc

    def _wrap(c):
        # For each field declared on *this class only* (not inherited),
        # replace MISSING defaults with None so plain @dataclass works.
        own_annotations = c.__dict__.get("__annotations__", {})
        for fname in list(own_annotations):
            val = c.__dict__.get(fname, _dc.MISSING)
            if val is _dc.MISSING:
                setattr(c, fname, None)
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

_stub_modules = {
    "isaaclab": types.ModuleType("isaaclab"),
    "isaaclab.utils": _isaaclab_utils,
    "isaaclab.envs": types.ModuleType("isaaclab.envs"),
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
