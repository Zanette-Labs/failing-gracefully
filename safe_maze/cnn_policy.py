import numpy as np
import torch
import torch.nn as nn
from safe_maze.env import SafeMazeEnv
from safe_maze.config import H, W


GRID_H = 2 * H - 1
GRID_W = 2 * W - 1


def obs_to_vec(env: SafeMazeEnv) -> np.ndarray:
    """
    CNN-friendly Markov observation.

    Shape:
        [7, GRID_H, GRID_W]

    Channels:
        0 = maze layout
        1 = goal position
        2 = agent position
        3 = steps remaining, broadcast everywhere
        4 = normalized row coordinate
        5 = normalized col coordinate
        6 = safe_reward, broadcast everywhere
    """
    GH, GW = env.grid.shape
    obs = np.zeros((7, GH, GW), dtype=np.float32)

    obs[0] = env.grid

    gr, gc = env.goal
    ar, ac = env.agent_pos

    obs[1, 2 * gr, 2 * gc] = 1.0
    obs[2, 2 * ar, 2 * ac] = 1.0

    steps_remaining = (env.max_steps - env._steps) / env.max_steps
    obs[3].fill(steps_remaining)

    rows = np.linspace(-1.0, 1.0, GH, dtype=np.float32)
    cols = np.linspace(-1.0, 1.0, GW, dtype=np.float32)

    obs[4] = rows[:, None]
    obs[5] = cols[None, :]

    obs[6].fill(getattr(env, '_obs_safe_reward', env.safe_reward))

    return obs


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class MazeResNetPolicy(nn.Module):
    """
    CNN-ResNet policy for fully observable mazes.

    Input:
        [B, 7, GRID_H, GRID_W]

    Output:
        action logits over 5 actions
    """

    def __init__(
        self,
        input_channels: int = 7,
        n_actions: int = 5,
        channels: int = 64,
        n_blocks: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(),
            *[ResidualBlock(channels) for _ in range(n_blocks)],
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, GRID_H, GRID_W)
            flat_dim = self.encoder(dummy).reshape(1, -1).shape[1]

        self.policy = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits [B, n_actions].
        """
        z = self.encoder(x)
        z = z.reshape(z.shape[0], -1)
        return self.policy(z)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        device = next(self.parameters()).device
        x = torch.from_numpy(obs).unsqueeze(0).to(device)

        logits = self(x)

        if deterministic:
            return int(logits.argmax(dim=-1).item())

        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())