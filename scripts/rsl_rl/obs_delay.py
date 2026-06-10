import gymnasium as gym
import torch


class DelayedObsWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=2):
        super().__init__(env)
        self.max_delay = max_delay
        self._num_envs = env.unwrapped.num_envs
        self._device = env.unwrapped.device
        self._delay = None
        self._buf = None
        self._obs_dim = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._delay = None
        self._buf = None
        return obs, info

    def _init_buf(self, obs):
        if isinstance(obs, dict):
            pol = obs["policy"]
        else:
            pol = obs
        self._obs_dim = pol.shape[1]
        self._delay = torch.randint(0, self.max_delay + 1, (self._num_envs,), device=self._device)
        max_d = self._delay.max().item()
        self._buf = torch.zeros(self._num_envs, max_d + 1, self._obs_dim, device=self._device)

    def _extract_policy(self, obs):
        return obs["policy"] if isinstance(obs, dict) else obs

    def _replace_policy(self, obs, delayed):
        if isinstance(obs, dict):
            obs["policy"] = delayed
            return obs
        return delayed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._delay is None:
            self._init_buf(obs)

        pol = self._extract_policy(obs)

        self._buf[:, 1:] = self._buf[:, :-1].clone()
        self._buf[:, 0] = pol

        delayed = self._buf[torch.arange(self._num_envs, device=self._device), self._delay]
        obs = self._replace_policy(obs, delayed)

        done = terminated | truncated
        if done.any():
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            new_delay = torch.randint(0, self.max_delay + 1, (done_idx.shape[0],), device=self._device)
            self._delay[done_idx] = new_delay
            self._buf[done_idx] = 0.0

        return obs, reward, terminated, truncated, info
