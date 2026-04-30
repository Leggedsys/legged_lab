from isaaclab.utils import configclass

from legged_lab.tasks.manager_based.legged_lab.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class DogPPORunnerCfg(PPORunnerCfg):
    experiment_name = "dog_locomotion"
    max_iterations = 800
    num_steps_per_env = 32
    save_interval = 25

    def __post_init__(self):
        self.policy.init_noise_std = 0.2
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
