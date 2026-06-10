import gymnasium as gym
import torch


class ActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=2):
        super().__init__(env)
        self.max_delay = max_delay
        self._num_envs = env.unwrapped.num_envs
        self._device = env.unwrapped.device
        self._delay = None
        self._buf = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._delay = None
        self._buf = None
        return obs, info

    def _init_buf(self, action_dim):
        self._delay = torch.randint(0, self.max_delay + 1, (self._num_envs,), device=self._device)
        self._buf = torch.zeros(self._num_envs, self.max_delay + 1, action_dim, device=self._device)

    def step(self, actions):
        if self._delay is None:
            self._init_buf(actions.shape[1])

        self._buf[:, 1:] = self._buf[:, :-1].clone()
        self._buf[:, 0] = actions

        delayed = self._buf[torch.arange(self._num_envs, device=self._device), self._delay]

        obs, reward, terminated, truncated, info = self.env.step(delayed)

        done = terminated | truncated
        if done.any():
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            new_delay = torch.randint(0, self.max_delay + 1, (done_idx.shape[0],), device=self._device)
            self._delay[done_idx] = new_delay
            self._buf[done_idx] = 0.0

        return obs, reward, terminated, truncated, info
