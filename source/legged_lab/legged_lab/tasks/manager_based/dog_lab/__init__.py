import gymnasium as gym

from . import agents


gym.register(
    id="Dog-Walk-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:PPORunnerCfg",
    },
)
