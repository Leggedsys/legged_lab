import torch
from rsl_rl.modules import ActorCritic

from legged_lab.tasks.manager_based.dog_lab.mdp.symmetry import _mirror_obs, _mirror_joint_group


class SymmetricActorCritic(ActorCritic):
    """Strictly equivariant actor via Reynolds averaging.

    For any obs o:
        π_sym(o) = ( π_θ(o) + m_a⁻¹( π_θ(m_o(o)) ) ) / 2

    Proof of equivariance:
        π_sym(m_o(o))
        = ( π_θ(m_o(o)) + m_a⁻¹(π_θ(m_o(m_o(o)))) ) / 2
        = ( π_θ(m_o(o)) + m_a⁻¹(π_θ(o)) ) / 2
        = m_a( π_sym(o) )   ✓

    The critic is left as a standard MLP (no symmetry constraint needed for training).
    """

    def _sym_mean(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.actor(observations)
        mean_m = _mirror_joint_group(self.actor(_mirror_obs(observations)))
        return (mean + mean_m) * 0.5

    def update_distribution(self, observations):
        mean = self._sym_mean(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        from torch.distributions import Normal
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        return self._sym_mean(observations)
