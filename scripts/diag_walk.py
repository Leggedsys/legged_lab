"""诊断脚本：打印实际加载后的 Walk 配置和一轮的 reward 分量"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import legged_lab.tasks  # noqa: F401  — 注册 gym task
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("Dog-Legged-Lab-Walk-v0", num_envs=4, device="cuda:0")

# 打印配置
print("=" * 60)
print("[CONFIG] base_lin_vel in obs:", hasattr(env_cfg.observations.policy, "base_lin_vel"))
print("[CONFIG] feet_air_time func:", env_cfg.rewards.feet_air_time.func.__module__ + "." + env_cfg.rewards.feet_air_time.func.__name__)
print("[CONFIG] alive weight:", env_cfg.rewards.alive.weight)
print(f"[CONFIG] track_lin_vel_xy: weight={env_cfg.rewards.track_lin_vel_xy_exp.weight}, "
      f"std={env_cfg.rewards.track_lin_vel_xy_exp.params.get('std')}")
print("[CONFIG] Commands ranges:")
print(f"  lin_vel_x={env_cfg.commands.base_velocity.ranges.lin_vel_x}")
print(f"  lin_vel_y={env_cfg.commands.base_velocity.ranges.lin_vel_y}")
print(f"  ang_vel_z={env_cfg.commands.base_velocity.ranges.ang_vel_z}")
print(f"  rel_standing={env_cfg.commands.base_velocity.rel_standing_envs}")
print("[CONFIG] init_noise_std:", env_cfg.observations.policy.enable_corruption)
print("=" * 60)

from isaaclab.envs import ManagerBasedRLEnv
env = ManagerBasedRLEnv(cfg=env_cfg)

obs, _ = env.reset()

try:
    for step in range(args_cli.steps):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        result = env.step(actions)
        obs, rewards, terminated, truncated, info = result

        if step == 0 or step == 19:
            msg = f"\n[step {step}]\n"
            for key in sorted(info.keys()):
                if key.startswith("rew_"):
                    msg += f"  {key}: {info[key][0].item():.4f}\n"
            cmd = env.command_manager.get_command("base_velocity")
            msg += f"  cmd vx: {cmd[0, 0].item():.3f}  vy: {cmd[0, 1].item():.3f}  wz: {cmd[0, 2].item():.3f}\n"
            lin_vel = env.scene["robot"].data.root_lin_vel_b[0].cpu().numpy()
            msg += f"  root lin_vel (body): {lin_vel}\n"
            print(msg, flush=True)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

env.close()
simulation_app.close()
