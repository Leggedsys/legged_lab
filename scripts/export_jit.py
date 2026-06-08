"""Export a RSL-RL checkpoint to TorchScript JIT for deployment."""

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output", type=str, default=None)
args_cli = parser.parse_args()

checkpoint_path = args_cli.checkpoint
output_path = args_cli.output or checkpoint_path.replace(".pt", "_jit.pt")

print(f"[INFO] Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

state = checkpoint.get("model_state_dict")
if state is None:
    raise ValueError("No model_state_dict in checkpoint")

obs_dim = state["actor.0.weight"].shape[1]
act_dim = state["actor.6.weight"].shape[0]
hidden_dims = [s for k, s in sorted((k, state[k].shape[0]) for k in state if "actor." in k and k.endswith(".weight"))][:-1]

print(f"[INFO] obs_dim={obs_dim}, act_dim={act_dim}, hidden_dims={hidden_dims}")


class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.ELU())
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.actor = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


model = PolicyNetwork(obs_dim, act_dim, hidden_dims)
actor_state = {k: v for k, v in state.items() if k.startswith("actor.")}
model.load_state_dict(actor_state)
model.eval()

example = torch.randn(1, obs_dim)
traced = torch.jit.trace(model, example)

with torch.no_grad():
    out1 = model(example)
    out2 = traced(example)
    assert torch.allclose(out1, out2, atol=1e-6), "JIT output mismatch!"
    print(f"[INFO] JIT verification passed. Output: {out1[0, :4].tolist()}")

traced.save(output_path)
print(f"[INFO] Exported: {output_path}")

meta = {"obs_dim": obs_dim, "act_dim": act_dim, "hidden_dims": hidden_dims, "joint_order": ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]}
torch.save(meta, output_path.replace(".pt", "_meta.pt"))
print(f"[INFO] Metadata: {output_path.replace('.pt', '_meta.pt')}")
