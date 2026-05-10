import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def foot_contact_balance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tau: float = 2.0,
) -> torch.Tensor:
    """Penalize uneven per-foot contact fraction via exponential moving average.

    Maintains a per-foot EMA of contact state (1=on ground, 0=in air).
    Penalises variance across the 4 feet.

    Unlike prolonged_air_penalty, this cannot be gamed by brief touches:
    a single-step contact moves the EMA by only alpha ≈ 0.01, so a foot
    that is up 99% of the time stays near 0 regardless.
    """
    from isaaclab.sensors import ContactSensor

    sensor: ContactSensor = env.scene[sensor_cfg.name]

    # 1 if currently in contact, 0 if in air
    in_contact = (sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0).float()

    alpha = env.step_dt / tau

    if not hasattr(env, "_contact_balance_ema"):
        env._contact_balance_ema = torch.full(
            (env.num_envs, len(sensor_cfg.body_ids)),
            0.5,
            device=env.device,
            dtype=torch.float32,
        )

    # Reset EMA for environments that just started a new episode
    env._contact_balance_ema[env.episode_length_buf == 1] = 0.5

    env._contact_balance_ema = alpha * in_contact + (1.0 - alpha) * env._contact_balance_ema

    # Return positive variance — caller applies negative weight to make this a penalty
    return torch.var(env._contact_balance_ema, dim=1)


def joint_power_distribution(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    惩罚各关节功率分布不均匀。
    功率 P_i = |τ_i| * |θ̇_i|，惩罚 var(P) 的平方。
    
    三条腿步态时：悬空腿 P≈0，支撑腿 P 极大 → 方差高 → 惩罚大。
    四条腿均匀步态：各腿功率接近 → 方差小 → 惩罚小。
    """
    # shape: (num_envs, num_joints)
    torques = env.scene["robot"].data.applied_torque
    joint_vel = env.scene["robot"].data.joint_vel

    power = torch.abs(torques) * torch.abs(joint_vel)          # (N, 12)
    return torch.var(power, dim=1)                              # (N,)


def foot_prolonged_air_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.5,
) -> torch.Tensor:
    """Terminate if any foot has been continuously airborne longer than threshold seconds.

    Hard constraint that makes sustained 3-leg gait infeasible regardless of reward shaping.
    Normal trot swing phase is ~0.2-0.3s, so threshold=1.5s never fires during normal gait.
    """
    from isaaclab.sensors import ContactSensor

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    current_air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]  # (N, 4)
    return (current_air_time > threshold).any(dim=1)  # (N,) bool


def pose_similarity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize joint position deviation from the default (standing) configuration.

    Returns the sum of squared differences between current and default joint positions.
    Apply with a negative weight to use as a penalty.
    """
    asset = env.scene["robot"]
    diff = asset.data.joint_pos - asset.data.default_joint_pos  # (N, 12)
    return torch.sum(torch.square(diff), dim=1)  # (N,)


def stand_still_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint deviation from default pose only when velocity command is near zero.

    At zero-velocity command the robot should hold the default standing pose.
    Returns pose L2 error gated by a binary mask (cmd_norm < threshold).
    Apply with a negative weight.
    """
    cmd = env.command_manager.get_command(command_name)  # (N, 3): [vx, vy, yaw]
    cmd_norm = torch.norm(cmd, dim=1)                    # (N,)
    is_still = (cmd_norm < velocity_threshold).float()   # 1 when standing, 0 when moving

    asset = env.scene["robot"]
    diff = asset.data.joint_pos - asset.data.default_joint_pos  # (N, 12)
    pose_err = torch.sum(torch.square(diff), dim=1)              # (N,)

    return is_still * pose_err


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.06,
) -> torch.Tensor:
    """Penalize feet below target_height during swing phase (DreamWaQ-style).

    Gated by horizontal foot velocity: stance feet (low v_xy) contribute ~0 naturally.
    Only penalizes downward error (foot below target), not overshoot.
    Apply with negative weight.
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]          # (N, 4)
    foot_vel_xy = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2   # (N, 4)
    )
    height_err_sq = torch.square(torch.clamp(target_height - foot_z, min=0.0))
    return torch.sum(height_err_sq * foot_vel_xy, dim=1)               # (N,)


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action difference (jerk): (a_t - 2·a_{t-1} + a_{t-2})².

    Apply with negative weight.
    """
    curr = env.action_manager.action       # a_t
    prev = env.action_manager.prev_action  # a_{t-1}

    if not hasattr(env, "_smooth_prev2"):
        env._smooth_prev2 = torch.zeros_like(curr)

    env._smooth_prev2[env.episode_length_buf == 1] = 0.0

    jerk = curr - 2.0 * prev + env._smooth_prev2
    env._smooth_prev2 = prev.clone()

    return torch.sum(torch.square(jerk), dim=1)