import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.dirname(__file__)


DOG_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSET_DIR, "dog", "dog_urdf.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    # NOTE: 左右 hip 关节在 URDF 中 axis 方向相反（FR/RR axis="-1 0 0", FL/RL axis="1 0 0"），
    # 但 URDF→USD 转换时 axis 的符号信息丢失，导致 USD 中所有 hip 的 axis 都变成正方向。
    # 因此 FR/RR hip 的值需要取反，才能在仿真中得到正确的向外摆腿姿态。
    # 此值与 URDF 的约定不一致，是 Isaac Lab 侧的适配。部署到真实机器人时需反向映射。
    # hip 默认值设为 0.0（中立位），避免训练时策略需要主动往回收腿才能竖直站立。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),
        joint_pos={
            # hip（外摆为正，FR/RR 取反；0.0 = 中立竖直站姿）
            "^FL_hip_joint$":    0.0,
            "^FR_hip_joint$":    0.0,
            "^RL_hip_joint$":    0.0,
            "^RR_hip_joint$":    0.0,
            # thigh（前摆为正）
            "^FL_thigh_joint$":  0.7,
            "^FR_thigh_joint$":  0.7,
            "^RL_thigh_joint$":  0.7,
            "^RR_thigh_joint$":  0.7,
            # calf（弯曲为正）
            "^FL_calf_joint$":  -1.2,
            "^FR_calf_joint$":  -1.2,
            "^RL_calf_joint$":  -1.2,
            "^RR_calf_joint$":  -1.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "front_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "^(FL|FR)_hip_joint$",
                "^(FL|FR)_thigh_joint$",
                "^(FL|FR)_calf_joint$",
            ],
            stiffness=60.0,
            damping=3.0,
        ),
        "rear_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "^(RL|RR)_hip_joint$",
                "^(RL|RR)_thigh_joint$",
                "^(RL|RR)_calf_joint$",
            ],
            stiffness=60.0,
            damping=3.0,
        ),
    },
)

# Isaac Lab ↔ URDF 关节符号映射
# 正值表示：hip 向外摆，thigh 向前摆，calf 弯曲
# 部署到真实机器人时，policy 输出乘以此映射即可还原 URDF 空间的值
DOG_JOINT_SIGN = {
    "FL_hip_joint":   1.0,
    "FR_hip_joint":  -1.0,
    "RL_hip_joint":  -1.0,
    "RR_hip_joint":   1.0,
    "FL_thigh_joint": 1.0,
    "FR_thigh_joint": 1.0,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint":  1.0,
    "FR_calf_joint":  1.0,
    "RL_calf_joint":  1.0,
    "RR_calf_joint":  1.0,
}
