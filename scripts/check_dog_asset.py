"""Visualize the dog asset without training."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize the dog asset.")
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Number of simulation steps before exiting. 0 means run until closed.",
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
    robot_cfg.spawn.func("/World/Robot_1", robot_cfg.spawn, translation=(0.0, 0.0, 0.2))
    robot = Articulation(robot_cfg.replace(prim_path="/World/Robot.*"))

    sim.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos, robot.data.default_joint_vel
    )
    robot.reset()

    print("[INFO] Dog asset loaded.")
    print("[INFO] Joint names:", robot.joint_names)
    print("[INFO] Default joint pos:", robot.data.default_joint_pos[0].cpu().numpy())

    step_count = 0
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        step_count += 1

        if step_count in (1, 10, 50, 100):
            print(
                f"[step {step_count}] joint pos: {robot.data.joint_pos[0].cpu().numpy()}"
            )
        if args_cli.steps > 0 and step_count >= args_cli.steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
