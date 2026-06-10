from legged_lab.tasks.manager_based.legged_lab.mdp import *  # noqa: F401, F403
from legged_lab.tasks.manager_based.legged_lab.mdp.rewards import (  # noqa: F401
    joint_pos_target_l2,
    track_height_exp,
    foot_slip,
    prolonged_air_penalty,
    trot_gait_reward,
    gait_clock_reward,
    gait_phase_obs,
    flat_orientation_exp,
)

from legged_lab.tasks.manager_based.dog_lab.mdp.commands import (  # noqa: F401
    UniformVelocityHeightCommand,
    UniformVelocityHeightCommandCfg,
)

from legged_lab.tasks.manager_based.dog_lab.mdp.rewards import (  # noqa: F401
    joint_power_distribution,
    foot_prolonged_air_termination,
    foot_contact_balance,
    pose_similarity_reward,
    stand_still_penalty,
    foot_clearance_reward,
    action_smoothness_l2,
    standup_height_delta,
    standup_upright_bonus,
)

from legged_lab.tasks.manager_based.dog_lab.mdp.symmetry import mirror_obs_action  # noqa: F401
