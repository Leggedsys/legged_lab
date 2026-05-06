from isaaclab.utils import configclass

from legged_lab.tasks.manager_based.dog_lab.agents.rsl_rl_walk_v2_ppo_cfg import DogWalkV2PPORunnerCfg


@configclass
class DogWalkV2Phase3PPORunnerCfg(DogWalkV2PPORunnerCfg):
    experiment_name = "dog_locomotion_walk_v2_phase3"
    max_iterations = 5000
    save_interval = 500
