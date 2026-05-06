from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class WalkPPORunnerCfg(PPORunnerCfg):
    experiment_name = "template_legged_lab_walk"
