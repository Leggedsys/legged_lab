"""设置默认关节角度，直接写入状态（无PD），测量身体高度和足端位置。"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure dog body height at default pose.")
parser.add_argument("--base-height", type=float, default=0.28)
parser.add_argument("--steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg

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

    # 无PD：stiffness/damping=0，只靠直接写关节状态维持位姿
    robot_cfg = DOG_URDF_CFG.copy()
    for name in robot_cfg.actuators:
        robot_cfg.actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=robot_cfg.actuators[name].joint_names_expr,
            stiffness=0.0,
            damping=0.0,
        )

    robot_cfg.spawn.func(
        "/World/Robot_1",
        robot_cfg.spawn,
        translation=(0.0, 0.0, args_cli.base_height),
    )
    robot = Articulation(robot_cfg.replace(prim_path="/World/Robot.*"))

    sim.reset()

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 2] = args_cli.base_height

    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(
        default_joint_pos, robot.data.default_joint_vel.clone()
    )
    robot.reset()

    print("Joint names:", robot.joint_names)
    print("Default joint pos:", default_joint_pos[0].cpu().numpy())
    print("Base spawn height:", args_cli.base_height)

    step_count = 0
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        robot.write_root_pose_to_sim(default_root_state[:, :7])
        robot.write_joint_state_to_sim(
            default_joint_pos, robot.data.default_joint_vel.clone()
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        step_count += 1

        if step_count in (1, 5, 10, 50, 100):
            root_z = robot.data.root_pos_w[0, 2].item()
            print(f"[step {step_count:3d}] base z = {root_z:.4f}")
            # 打印足端高度
            foot_names = [n for n in robot.body_names if "_foot" in n]
            if foot_names:
                foot_ids = [robot.body_names.index(n) for n in foot_names]
                body_pos = robot.data.body_pos_w[0, foot_ids]
                print(f"  feet z: {body_pos[:, 2].cpu().numpy()}")

        if args_cli.steps > 0 and step_count >= args_cli.steps:
            break

    print(f"Final base height: {robot.data.root_pos_w[0, 2].item():.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
