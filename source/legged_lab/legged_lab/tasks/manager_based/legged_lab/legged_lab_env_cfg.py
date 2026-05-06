# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from legged_lab.assets.dog_cfg import DOG_URDF_CFG


##
# Scene
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # 先用平地，后续加复杂地形
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # 机器人
    robot = DOG_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# Actions
##


@configclass
class ActionsCfg:
    # 前腿关节目标位置（相对于默认站姿的偏移）
    front_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "^(FL|FR)_hip_joint$",
            "^(FL|FR)_thigh_joint$",
            "^(FL|FR)_calf_joint$",
        ],
        scale={
            ".*_hip_joint": 0.025,
            ".*_thigh_joint": 0.035,
            ".*_calf_joint": 0.02,
        },
        use_default_offset=True,
    )

    # 后腿关节目标位置（相对于默认站姿的偏移）
    rear_legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "^(RL|RR)_hip_joint$",
            "^(RL|RR)_thigh_joint$",
            "^(RL|RR)_calf_joint$",
        ],
        scale={
            ".*_hip_joint": 0.03,
            ".*_thigh_joint": 0.045,
            ".*_calf_joint": 0.025,
        },
        use_default_offset=True,
    )


##
# Observations
##


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 机体角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        # 投影重力向量
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # 速度指令
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # 关节位置（相对默认站姿）
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        # 关节速度
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 上一步动作
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events（Domain Randomization）
##


@configclass
class EventCfg:
    # 启动时随机化物理参数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 启动时随机化关节刚度阻尼
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset时随机化基座状态
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.98, 1.02),
            "velocity_range": (0.0, 0.0),
        },
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    # 存活奖励
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # 速度跟踪（线速度xy）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 速度跟踪（角速度z）
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 惩罚z方向线速度
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)

    # 惩罚机身倾倒带来的姿态晃动
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 惩罚关节力矩（节能）
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # 惩罚关节加速度
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)

    # 惩罚动作变化率
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

    # 保持默认站姿，降低训练早期的搜索难度
    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.06)

    # 约束机身高度，避免通过趴地获得局部最优
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-1.5, params={"target_height": 0.355}
    )

    # 惩罚不期望的接触
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 1.0,
        },
    )

    # 惩罚摔倒
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


##
# Terminations
##


@configclass
class TerminationsCfg:
    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 机体接触地面
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )

    # 倾倒或过低时尽快结束，减少无效样本
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.20}
    )


##
# Commands
##


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


##
# Curriculum
##


@configclass
class CurriculumCfg:
    # 先学稳定站立，之后再逐步放开前进跟踪
    enable_lin_vel_tracking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "track_lin_vel_xy_exp", "weight": 1.5, "num_steps": 50000},
    )

    relax_joint_deviation = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation", "weight": -0.02, "num_steps": 50000},
    )

    relax_base_height = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "base_height", "weight": -0.8, "num_steps": 50000},
    )

    enable_forward_command = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "command_manager.cfg.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.enable_command_range,
            "modify_params": {"target_value": (0.0, 0.6), "num_steps": 50000},
        },
    )


##
# Environment
##


@configclass
class LeggedLabEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4  # 控制频率 = 1/(dt*decimation) = 50Hz
        self.episode_length_s = 20.0
        self.sim.dt = 0.005  # 200Hz物理仿真
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # PD参数
        self.scene.robot.actuators["front_legs"].stiffness = 18.0
        self.scene.robot.actuators["front_legs"].damping = 1.0
        self.scene.robot.actuators["rear_legs"].stiffness = 32.0
        self.scene.robot.actuators["rear_legs"].damping = 2.2


@configclass
class LeggedLabWalkEnvCfg(LeggedLabEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # locomotion阶段：直接启用低速前进命令与速度跟踪
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.joint_deviation.weight = -0.02
        self.rewards.base_height.weight = -0.8

        # 行走任务不再需要站立到前进的课程切换
        self.curriculum = None
