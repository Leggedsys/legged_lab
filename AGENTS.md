# AGENTS.md — legged_lab

## Overview
Isaac Lab extension for legged-robot RL using RSL-RL (PPO). Requires Isaac Sim (4.5–5.0).
Shared MDP code lives in `legged_lab/`, dog-specific overrides in `dog_lab/`.

## Install
```bash
# Requires Isaac Lab installed separately. Use isaaclab's python, not system python.
python -m pip install -e source/legged_lab
```

## Run

```bash
# List registered envs
python scripts/list_envs.py

# Train / play (use isaaclab python.sh wrapper if not in conda)
python scripts/rsl_rl/train.py --task=Dog-Walk-v1
python scripts/rsl_rl/play.py --task=Dog-Walk-v1

# WTW energy-adaptive walking
python scripts/rsl_rl/train.py --task=Dog-WTW-v0
python scripts/rsl_rl/play.py --task=Dog-WTW-v0

# Dummy agents (verify env wiring)
python scripts/zero_agent.py --task=Template-Legged-Lab-v0
python scripts/random_agent.py --task=Dog-Walk-v1
```

`list_envs.py` hardcodes a search pattern (`Dog-Legged-Lab` / `Template-Legged-Lab`). If task names
change, update the pattern in that file or the listed envs won't appear.

## Test

Unit tests run **without** Isaac Sim via stub injection in `tests/conftest.py`.
```bash
python -m pytest tests/ -v
python -m pytest tests/test_rewards.py -v
```

## Lint / format
```bash
pre-commit run --all-files
```
Uses: black (line-length 120, `--unstable`), flake8 (google-docstrings, ignores E402/E501/D401), isort (black profile), pyupgrade (py310+), codespell, insert-license (BSD-3-Clause).

flake8 per-file ignores *only* `*/__init__.py:F401`; import-usage errors elsewhere are enforced.

## Architecture

```
source/legged_lab/legged_lab/tasks/manager_based/
├── legged_lab/                         # Shared MDP, base env config, agents
│   ├── __init__.py                     # Registers Template-Legged-Lab-v0, Template-Legged-Lab-Walk-v0
│   ├── legged_lab_env_cfg.py           # Base LeggedLabEnvCfg, LeggedLabWalkEnvCfg
│   ├── mdp/
│   │   ├── __init__.py                 # Re-exports all shared MDP symbols
│   │   ├── rewards.py                  # Core reward implementations (_impl + wrapper)
│   │   └── curriculums.py
│   └── agents/
│       ├── rsl_rl_ppo_cfg.py           # Base PPORunnerCfg
│       └── rsl_rl_walk_ppo_cfg.py      # WalkPPORunnerCfg
├── dog_lab/                            # Dog-specific env configs, terrains, commands, rewards
│   ├── __init__.py                     # Registers Dog-Walk-v1, Dog-Walk-Terrain-v1
│   ├── dog_env_cfg.py                  # DogEnvCfg, DogWalkEnvCfg, DogWalkTerrainEnvCfg, etc.
│   ├── symmetric_actor_critic.py       # SymmetricActorCritic for symmetry-based RL
│   ├── terrains.py                     # Competition & flat terrain configs
│   ├── mdp/
│   │   ├── __init__.py                 # from legged_lab.mdp import * + dog-specific overrides
│   │   ├── commands.py                 # UniformVelocityHeightCommand + Cfg
│   │   ├── rewards.py                  # Dog-specific reward functions
│   │   └── symmetry.py                 # mirror_obs_action
│   └── agents/
│       └── ppo_cfg.py                  # PPORunnerCfg, TerrainPPORunnerCfg (extends base)
├── dog_wtw/                            # WTW energy-adaptive walking
│   ├── __init__.py                     # Registers Dog-WTW-v0, Dog-WTW-Terrain-v0
│   ├── wtw_env_cfg.py                  # WtwRLEnv, WtwEnvCfg, WtwTerrainEnvCfg
│   ├── wtw_reward_manager.py           # Ji22RewardManager (multiplicative pos×exp(neg/σ))
│   ├── mdp/
│   │   ├── __init__.py                 # from dog_lab.mdp import * + energy_new_actual
│   │   └── rewards.py                  # _energy_new_actual_impl + wrapper
│   └── agents/
│       └── wtw_ppo_cfg.py              # WtwPPORunnerCfg, WtwTerrainPPORunnerCfg
└── assets/dog/                         # URDF and USD assets
```

- `dog_lab/__init__.py` sets `rsl_rl_cfg_entry_point` to `agents.ppo_cfg:PPORunnerCfg` (not `rsl_rl_*`).
- `dog_wtw/` uses a custom `WtwRLEnv` subclass that swaps in `Ji22RewardManager` for multiplicative
  reward clipping — reward = Σ(pos) × exp(Σ(neg) / 0.02).
- `legged_lab/__init__.py` and `dog_lab/__init__.py` register gym envs directly. The top-level
  `tasks/__init__.py` uses `isaaclab_tasks.utils.import_packages` to recursively import sub-packages
  and trigger those registrations.
- `dog_lab/mdp/__init__.py` does `from legged_lab.tasks.manager_based.legged_lab.mdp import *`
  then adds dog-specific symbols (`UniformVelocityHeightCommandCfg`, `foot_clearance_reward`, etc.).

## Conventions
- Line length: 120 (black + flake8).
- Python 3.10+ (pyupgrade `--py310-plus`).
- BSD-3-Clause license header required on all `.py` and `.yaml` files (pre-commit insert-license).
- No USD files in repo (gitignored); large binary assets use git-lfs (`.pt`, `.usd`, etc.).
- Reward functions expose a pure-tensor `_impl` variant for testability.
- Google-style docstrings (flake8 `docstring-convention=google`).

### Agent naming conventions
- `legged_lab/agents/` uses `rsl_rl_*.py` prefix.
- `dog_lab/agents/` uses **no** prefix — just `ppo_cfg.py`.
- `dog_lab/agents/__init__.py` is **empty**. This is intentional; do not add imports there.

### MDP module import pattern
`dog_lab/mdp/__init__.py` explicitly re-exports symbols from both the shared `legged_lab.mdp` and
its own sub-modules. When adding a new reward/command/symmetry, add the corresponding `from ... import`
line there so `dog_env_cfg.py` can reference it via `mdp.func_name`.

## Gotchas
- **Must use Isaac Sim python** to run scripts — the package imports `isaaclab.*` and `omni.*`
  which only exist inside the sim runtime.
- Tests avoid this via `tests/conftest.py`: it injects stub `sys.modules` for `omni`, `isaaclab.*`,
  and `isaaclab_rl.*`, then loads source files directly with `importlib`. Adding new `isaaclab`
  imports to tested source files may require updating the conftest stubs.
- Asset caching: `.isaac_cache/` directory is generated.
- `.gitignore` forbids USD files; `.gitattributes` marks them (plus `.pt`, `.jit`, `.mp4`, etc.) for git-lfs.
