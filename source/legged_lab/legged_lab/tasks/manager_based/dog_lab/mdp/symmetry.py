import torch

# Observation layout for DogWalkEnvCfg (concatenated order):
#   base_lin_vel      [0:3]   vx, vy, vz
#   base_ang_vel      [3:6]   wx, wy, wz
#   projected_gravity [6:9]   gx, gy, gz
#   velocity_commands [9:12]  vx_cmd, vy_cmd, yaw_cmd
#   joint_pos         [12:24] 12 joints
#   joint_vel         [24:36] 12 joints
#   actions           [36:48] 12 joints
#   height_scan       [48:373] 13 y-rows × 25 x-cols = 325

# Joint order: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#               RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
# Mirror: swap FL(0-2)↔FR(3-5), RL(6-8)↔RR(9-11)
_JOINT_MIRROR_IDX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# FL/RL hips have axis +X, FR/RR hips have axis -X → negate all hips after swap
_JOINT_MIRROR_SIGN = torch.tensor([-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], dtype=torch.float32)

# Height scan grid: outer loop = y (13 rows), inner loop = x (25 cols)
_H_ROWS = 13
_H_COLS = 25


def _mirror_joint_group(j: torch.Tensor) -> torch.Tensor:
    sign = _JOINT_MIRROR_SIGN.to(j.device)
    return j[:, _JOINT_MIRROR_IDX] * sign


def _mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    m = obs.clone()
    # base_lin_vel: negate vy
    m[:, 1] = -obs[:, 1]
    # base_ang_vel: negate wx, wz
    m[:, 3] = -obs[:, 3]
    m[:, 5] = -obs[:, 5]
    # projected_gravity: negate gy
    m[:, 7] = -obs[:, 7]
    # velocity_commands: negate vy_cmd, yaw_cmd
    m[:, 10] = -obs[:, 10]
    m[:, 11] = -obs[:, 11]
    # joint groups: swap left↔right legs and negate hip signs
    m[:, 12:24] = _mirror_joint_group(obs[:, 12:24])
    m[:, 24:36] = _mirror_joint_group(obs[:, 24:36])
    m[:, 36:48] = _mirror_joint_group(obs[:, 36:48])
    # height scan: flip y-axis (dim 1 after batch dim)
    n = obs.shape[0]
    m[:, 48:] = obs[:, 48:].view(n, _H_ROWS, _H_COLS).flip(1).reshape(n, _H_ROWS * _H_COLS)
    return m


def mirror_obs_action(obs, actions, env, obs_type):
    """Left-right symmetry augmentation for DogWalkEnv.

    Returns the batch doubled: [original | mirrored] for both obs and actions.
    Either obs or actions may be None, in which case the corresponding return is None.
    """
    m_obs = _mirror_obs(obs) if obs is not None else None
    m_act = _mirror_joint_group(actions) if actions is not None else None

    aug_obs = torch.cat([obs, m_obs], dim=0) if obs is not None else None
    aug_act = torch.cat([actions, m_act], dim=0) if actions is not None else None

    return aug_obs, aug_act
