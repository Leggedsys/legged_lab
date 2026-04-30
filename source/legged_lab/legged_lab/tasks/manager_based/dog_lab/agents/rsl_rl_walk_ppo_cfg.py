from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import DogPPORunnerCfg


@configclass
class DogWalkPPORunnerCfg(DogPPORunnerCfg):
    experiment_name = "dog_locomotion_walk"
    max_iterations = 3000
    save_interval = 200
    empirical_normalization = True

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.25
