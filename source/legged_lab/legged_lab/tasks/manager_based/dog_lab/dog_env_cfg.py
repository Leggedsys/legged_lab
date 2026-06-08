import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from legged_lab.assets.dog_cfg import DOG_URDF_CFG
from legged_lab.tasks.manager_based.dog_lab.terrains import (
    COMPETITION_TERRAIN_IMPORTER_CFG,
    FLAT_TERRAIN_IMPORTER_CFG,
)

from . import mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = DOG_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.7, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.2, 0.6]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        scale={
            ".*_hip_joint": 0.15,
            ".*_thigh_joint": 0.20,
            ".*_calf_joint": 0.20,
        },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # 陀螺偏置: ±0.03 rad/s persistent drift, simulated per-episode bias
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                bias_noise_cfg=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
                sample_bias_per_component=True,
            ),
        )
        # 重力投影偏置: ±0.02 g persistent drift
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
                bias_noise_cfg=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
                sample_bias_per_component=True,
            ),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for Critic only (not available at deployment)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 40.0),
            "torque_range": (0.0, 5.0),
        },
    )

    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.2),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    randomize_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_thigh|.*_calf|.*_foot"),
            "mass_distribution_params": (-0.15, 0.15),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    # ─────────────────────────────────────────────
    # 1. 任务奖励  Task rewards
    # ─────────────────────────────────────────────

    # 线速度跟踪: exp(-‖v_cmd - v_xy‖² / σ²)
    # σ=0.25*v_cmd ≈ 0.25~0.5, 取0.5容忍初期较大跟踪误差
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 偏航角速度跟踪: exp(-|ω_cmd - ω|² / σ²)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 存活奖励: 不摔倒就给保底信号，鼓励探索期保持站立
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # ─────────────────────────────────────────────
    # 2. 惩罚项 — 稳定性  Stability penalties
    # ─────────────────────────────────────────────

    # 惩罚z向弹跳: 防止策略靠上下蹦"作弊"
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)

    # 惩罚机身偏离默认高度 0.30m
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-0.8, params={"target_height": 0.30})

    # 惩罚横滚/俯仰角速度
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 机身倾斜 — 指数形式: exp(-‖g_xy‖² / σ), σ=0.05
    # 5°→0.99, 10°→0.94, 20°→0.78, 45°→0.37
    # 用正权重: 奖励平坦姿态, 倾斜时指数衰减
    flat_orientation_exp = RewTerm(
        func=mdp.flat_orientation_exp,
        weight=0.2,
        params={"sigma": 0.05},
    )

    # ─────────────────────────────────────────────
    # 2. 惩罚项 — 效率与平滑  Efficiency & smoothness
    # ─────────────────────────────────────────────

    # L2力矩惩罚: 防止暴力解, 对应能耗
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)

    # 动作一阶差分: sim-to-real关键, 抑制高频抖动
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 动作二阶差分 (jerk): 更严格平滑, 减少电机冲击
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.015)

    # ─────────────────────────────────────────────
    # 2. 惩罚项 — 安全约束  Safety
    # ─────────────────────────────────────────────

    # 小腿触地: 防止膝盖跪地步态
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 1.0,
        },
    )

    # ─────────────────────────────────────────────
    # 2. 惩罚项 — 关节极限 (soft exponential, 替代 L1 偏离)
    # ─────────────────────────────────────────────

    # 接近物理极限时指数级惩罚, 远离时≈0
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ─────────────────────────────────────────────
    # 3. 先验项 — 步态质量  Gait quality priors
    # ─────────────────────────────────────────────

    # 对角步态同步 (trot): FL+RR / FR+RL 交替触地
    # 消除齐跳和滑步, 加速收敛到自然步态
    trot_gait = RewTerm(
        func=mdp.trot_gait_reward,
        weight=0.8,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    # 足端滑动惩罚: 支撑相足端水平速度 -> 驱动抬脚
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # 长时间悬空惩罚: 防止3腿/2腿步态
    prolonged_air = RewTerm(
        func=mdp.prolonged_air_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
        },
    )

    # 足端接触时间均衡: 惩罚某对脚踩地时间明显长于另一对
    foot_contact_balance = RewTerm(
        func=mdp.foot_contact_balance,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    # ─────────────────────────────────────────────
    # 3. 先验项 — 姿态保持  Pose priors
    # ─────────────────────────────────────────────

    # 足端离地高度不足惩罚 (DreamWaQ-style)
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.10,
        },
    )

    # 零速度指令时惩罚关节偏离
    stand_still = RewTerm(
        func=mdp.stand_still_penalty,
        weight=-0.1,
        params={"command_name": "base_velocity", "velocity_threshold": 0.1},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.08})
    # 任意一只脚连续离地超过 1.5 s 即终止：正常对角步态摆腿约 0.2-0.3 s，
    # 不会误触发；三条腿步态中悬空脚持续数秒，必然触发。
    prolonged_air = DoneTerm(
        func=mdp.foot_prolonged_air_termination,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class DogWalkEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 1: flat terrain."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.terrain = FLAT_TERRAIN_IMPORTER_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        self.curriculum = None


@configclass
class DogWalkTerrainEnvCfg(DogWalkEnvCfg):
    """Phase 2: 10-level competition terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = COMPETITION_TERRAIN_IMPORTER_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        self.curriculum = CurriculumCfg()
