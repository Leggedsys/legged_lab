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
            ".*_calf_joint": 0.15,
        },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
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
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 25.0),
            "torque_range": (0.0, 2.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
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
    # 任务奖励  Task rewards
    # ─────────────────────────────────────────────

    # std=0.5 → exp(-4·err²)，初期容忍更大的跟踪误差，梯度更平滑
    # std=0.25 适合已经会走之后的精调阶段
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ─────────────────────────────────────────────
    # 稳定性约束  Stability
    # ─────────────────────────────────────────────

    # 惩罚 z 向线速度：防止策略靠上下蹦来"作弊"速度奖励
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    # 惩罚横滚/俯仰角速度（动态）
    # 权重小于 flat_orientation 是合理的：姿态角比角速度更容易观测和约束
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 惩罚机身倾斜（静态），用重力向量投影，无万向锁问题
    # DreamWaQ 使用 -0.2；-1.0 过强，会阻止坡道上的躯干自适应前倾
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1)

    # ─────────────────────────────────────────────
    # 效率与平滑  Efficiency & smoothness
    # ─────────────────────────────────────────────

    # 惩罚力矩 L2：防止暴力解，对应能耗
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)

    # 惩罚动作一阶差分：直接对应真机控制信号的变化率，sim-to-real 关键项
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 惩罚动作二阶差分（jerk）：比一阶更严格的平滑约束，减少真机电机冲击
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)

    # ─────────────────────────────────────────────
    # 安全约束  Safety
    # ─────────────────────────────────────────────

    # 惩罚小腿接触地面：防止膝盖跪地步态
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 1.0,
        },
    )

    # ─────────────────────────────────────────────
    # 姿态保持  Pose similarity
    # ─────────────────────────────────────────────

    # 惩罚关节角度偏离默认站姿：防止策略让某条腿长期抬起或关节极端弯曲
    # 地形阶段降低权重，给坡道姿态适应留空间
    pose_similarity = RewTerm(
        func=mdp.pose_similarity_reward,
        weight=-0.02,
    )

    # 摆腿时脚底高度不足惩罚（DreamWaQ-style）：迫使策略主动抬脚，地形适应关键项
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.10,
        },
    )

    # 零速度指令时额外惩罚关节偏离：强制静止时保持默认站姿
    stand_still = RewTerm(
        func=mdp.stand_still_penalty,
        weight=-0.5,
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
