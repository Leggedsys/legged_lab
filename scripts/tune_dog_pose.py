"""Spawn the dog asset with a manually specified standing pose."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tune the dog standing pose.")
parser.add_argument("--base-height", type=float, default=0.34)
parser.add_argument("--fl-hip", type=float, default=0.10)
parser.add_argument("--fr-hip", type=float, default=-0.10)
parser.add_argument("--rl-hip", type=float, default=0.10)
parser.add_argument("--rr-hip", type=float, default=-0.10)
parser.add_argument("--front-thigh", type=float, default=0.30)
parser.add_argument("--rear-thigh", type=float, default=0.45)
parser.add_argument("--front-calf", type=float, default=-0.70)
parser.add_argument("--rear-calf", type=float, default=-0.85)
parser.add_argument("--steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from legged_lab.assets.dog_cfg import DOG_URDF_CFG


def main():
    sim = SimulationContext(
        physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0"
    )
    set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.3])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = DOG_URDF_CFG.copy()
    robot_cfg.spawn.func(
        "/World/Robot_1", robot_cfg.spawn, translation=(0.0, 0.0, args_cli.base_height)
    )
    robot = Articulation(robot_cfg.replace(prim_path="/World/Robot.*"))

    sim.reset()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_ids = {name: idx for idx, name in enumerate(robot.joint_names)}

    pose_map = {
        "FL_hip_joint": args_cli.fl_hip,
        "FR_hip_joint": args_cli.fr_hip,
        "RL_hip_joint": args_cli.rl_hip,
        "RR_hip_joint": args_cli.rr_hip,
        "FL_thigh_joint": args_cli.front_thigh,
        "FR_thigh_joint": args_cli.front_thigh,
        "RL_thigh_joint": args_cli.rear_thigh,
        "RR_thigh_joint": args_cli.rear_thigh,
        "FL_calf_joint": args_cli.front_calf,
        "FR_calf_joint": args_cli.front_calf,
        "RL_calf_joint": args_cli.rear_calf,
        "RR_calf_joint": args_cli.rear_calf,
    }
    for name, value in pose_map.items():
        joint_pos[:, joint_ids[name]] = value

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    print("[INFO] Using pose:")
    for name, value in pose_map.items():
        print(f"  {name}: {value}")

    step_count = 0
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        step_count += 1
        if args_cli.steps > 0 and step_count >= args_cli.steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
