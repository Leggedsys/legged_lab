import gymnasium as gym

from . import agents


gym.register(
    id="Dog-Legged-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DogPPORunnerCfg",
    },
)


gym.register(
    id="Dog-Legged-Lab-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_walk_ppo_cfg:DogWalkPPORunnerCfg",
    },
)


gym.register(
    id="Dog-Legged-Lab-Walk-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogWalkV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_walk_v2_ppo_cfg:DogWalkV2PPORunnerCfg",
    },
)


gym.register(
    id="Dog-Legged-Lab-Walk-v2-Phase1p5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogWalkV2Phase1p5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_walk_v2_phase1p5_ppo_cfg:DogWalkV2Phase1p5PPORunnerCfg",
    },
)


gym.register(
    id="Dog-Legged-Lab-Walk-v2-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dog_env_cfg:DogWalkV2Phase2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_walk_v2_phase2_ppo_cfg:DogWalkV2Phase2PPORunnerCfg",
    },
)
