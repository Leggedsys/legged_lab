"""Open-loop gait debugger for the dog asset."""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug open-loop dog joint motions.")
parser.add_argument(
    "--mode",
    type=str,
    default="rear_only",
    choices=["stand", "front_only", "rear_only", "trot"],
    help="Joint excitation pattern to play.",
)
parser.add_argument(
    "--amplitude", type=float, default=0.18, help="Sinusoid amplitude in rad."
)
parser.add_argument(
    "--frequency", type=float, default=1.2, help="Sinusoid frequency in Hz."
)
parser.add_argument(
    "--steps", type=int, default=600, help="Number of simulation steps before exiting."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from legged_lab.assets.dog_cfg import DOG_URDF_CFG


def build_targets(robot: Articulation, t: float) -> object:
    targets = robot.data.default_joint_pos.clone()
    a = args_cli.amplitude
    w = 2.0 * math.pi * args_cli.frequency
    s = math.sin(w * t)

    joint_ids = {name: idx for idx, name in enumerate(robot.joint_names)}

    def add_delta(name: str, delta: float):
        targets[:, joint_ids[name]] += delta

    def drive_leg(prefix: str, phase: float):
        leg_s = math.sin(w * t + phase)
        add_delta(f"{prefix}_thigh_joint", 0.7 * a * leg_s)
        add_delta(f"{prefix}_calf_joint", -1.0 * a * leg_s)

    if args_cli.mode == "front_only":
        drive_leg("FL", 0.0)
        drive_leg("FR", math.pi)
    elif args_cli.mode == "rear_only":
        drive_leg("RL", 0.0)
        drive_leg("RR", math.pi)
    elif args_cli.mode == "trot":
        drive_leg("FL", 0.0)
        drive_leg("RR", 0.0)
        drive_leg("FR", math.pi)
        drive_leg("RL", math.pi)
    else:
        _ = s

    return targets


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
        "/World/Robot_1", robot_cfg.spawn, translation=(0.0, 0.0, 0.33)
    )
    robot = Articulation(robot_cfg.replace(prim_path="/World/Robot.*"))

    sim.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos, robot.data.default_joint_vel
    )
    robot.reset()

    print(f"[INFO] Debug mode: {args_cli.mode}")
    print(f"[INFO] amplitude={args_cli.amplitude}, frequency={args_cli.frequency}")
    print("[INFO] Joint names:", robot.joint_names)

    t = 0.0
    dt = sim.get_physics_dt()
    step_count = 0
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        targets = build_targets(robot, t)
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        t += dt
        step_count += 1
        if step_count >= args_cli.steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
