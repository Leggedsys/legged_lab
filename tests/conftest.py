"""
conftest.py: Make pure-tensor reward helpers importable without Isaac Sim.

The legged_lab package __init__.py imports isaaclab_tasks which in turn
imports omni.log (only available inside the Isaac Sim runtime). This
conftest injects stub package entries into sys.modules so that the
individual rewards.py module can be loaded for unit-testing the pure-tensor
_impl functions without starting the simulator.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Path to the actual rewards.py source file
# ---------------------------------------------------------------------------
_REWARDS_FILE = (
    Path(__file__).parent.parent
    / "source/legged_lab/legged_lab/tasks/manager_based/legged_lab/mdp/rewards.py"
)

# ---------------------------------------------------------------------------
# Inject stub package hierarchy into sys.modules so that the dotted-path
# import `legged_lab.tasks.manager_based.legged_lab.mdp.rewards` resolves
# to the real file without executing any __init__.py that would pull in omni.
# ---------------------------------------------------------------------------
_PKG_CHAIN = [
    "legged_lab",
    "legged_lab.tasks",
    "legged_lab.tasks.manager_based",
    "legged_lab.tasks.manager_based.legged_lab",
    "legged_lab.tasks.manager_based.legged_lab.mdp",
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
