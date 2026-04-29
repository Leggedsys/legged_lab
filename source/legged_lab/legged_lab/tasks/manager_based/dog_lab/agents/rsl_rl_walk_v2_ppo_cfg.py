from isaaclab.utils import configclass

from legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_ppo_cfg import DogPPORunnerCfg


@configclass
class DogWalkV2PPORunnerCfg(DogPPORunnerCfg):
    experiment_name = "dog_locomotion_walk_v2"
    max_iterations = 5000
    num_steps_per_env = 24
    save_interval = 200
    empirical_normalization = True

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.01        # 10× base default — prevents premature convergence
        self.algorithm.learning_rate = 1e-3
        self.algorithm.schedule = "fixed"          # prevents LR→0 collapse seen in prior runs
