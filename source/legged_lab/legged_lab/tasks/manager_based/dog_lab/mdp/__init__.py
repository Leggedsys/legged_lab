from legged_lab.tasks.manager_based.legged_lab.mdp import *  # noqa: F401, F403
from legged_lab.tasks.manager_based.legged_lab.mdp.rewards import (  # noqa: F401
    joint_pos_target_l2,
    track_height_exp,
    foot_slip,
    prolonged_air_penalty,
    trot_gait_reward,
    gait_clock_reward,
    gait_phase_obs,
)

from legged_lab.tasks.manager_based.dog_lab.mdp.commands import (  # noqa: F401
    UniformVelocityHeightCommand,
    UniformVelocityHeightCommandCfg,
)
