# AGENTS.md

## Setup
- This repo is an Isaac Lab extension, not a standalone app. Use a Python environment that already has Isaac Lab installed, then install this package with `python -m pip install -e source/legged_lab`.
- If Isaac Lab is not installed into the active interpreter, run scripts via Isaac Lab's launcher (`python.sh` / `isaaclab.sh -p`) instead of plain `python`; the README calls this out and the scripts import `isaaclab.*` immediately.

## Repo Shape
- The only Python package in this repo is `source/legged_lab/legged_lab`.
- Importing `legged_lab` registers the task and UI extension from `source/legged_lab/legged_lab/__init__.py`.
- Task auto-discovery happens through `source/legged_lab/legged_lab/tasks/__init__.py`, which uses `isaaclab_tasks.utils.import_packages()` and skips `utils` and `.mdp` packages.

## Task Entrypoints
- The only registered gym task in this repo is `Template-Legged-Lab-v0`, defined in `source/legged_lab/legged_lab/tasks/manager_based/legged_lab/__init__.py`.
- Environment behavior is wired from `source/legged_lab/legged_lab/tasks/manager_based/legged_lab/legged_lab_env_cfg.py`.
- PPO runner defaults are in `source/legged_lab/legged_lab/tasks/manager_based/legged_lab/agents/rsl_rl_ppo_cfg.py`.
- `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` are the real train/play entrypoints; both rely on Hydra task config lookup from the gym registry, so `--task` must match the registered id exactly.

## Verified Commands
- List environments: `python scripts/list_envs.py`
- Train with RSL-RL: `python scripts/rsl_rl/train.py --task Template-Legged-Lab-v0`
- Play a checkpoint: `python scripts/rsl_rl/play.py --task Template-Legged-Lab-v0 --checkpoint <path>`
- Format/lint hook pass: `pre-commit run --all-files`

## Gotchas
- `scripts/list_envs.py` still filters task ids by the prefix `"Template-"`. If you rename the task id, update that script too or your env will not appear in the listing.
- Logs and checkpoints go under `logs/rsl_rl/<experiment_name>/...`; the current default `experiment_name` is `go2_locomotion`.
- `play.py` always exports the loaded policy to JIT and ONNX under `<checkpoint run dir>/exported/`.
- `train.py --distributed` requires `rsl-rl-lib>=2.3.1`; the script hard-fails with an install hint if the version is older.

## Local Conventions Worth Preserving
- This repo still contains upstream template metadata in several places (`Template-Legged-Lab-v0`, `Template-` filter, `Extension Template` in `extension.toml`). Treat renames as cross-file changes, not single-file edits.
- `legged_lab_env_cfg.py` currently contains Chinese inline comments; preserve the existing comment language/style in that file unless there is a reason to normalize it.
- There is no repo-local CI workflow, test suite, or dedicated typecheck command checked in here. Prefer focused script-level validation plus `pre-commit run --all-files` over inventing extra verification steps.
