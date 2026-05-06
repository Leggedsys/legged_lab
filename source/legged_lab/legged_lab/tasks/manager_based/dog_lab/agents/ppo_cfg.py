from isaaclab.utils import configclass

from legged_lab.tasks.manager_based.legged_lab.agents.rsl_rl_ppo_cfg import PPORunnerCfg as _BasePPORunnerCfg


@configclass
class PPORunnerCfg(_BasePPORunnerCfg):
    experiment_name = "dog_walk"
    max_iterations = 5000
    num_steps_per_env = 24
    save_interval = 200
    empirical_normalization = True

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.01
        self.algorithm.learning_rate = 3e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01
        self.algorithm.max_grad_norm = 0.5
