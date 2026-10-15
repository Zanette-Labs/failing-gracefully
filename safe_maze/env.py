import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple, Dict, Any


# Action constants  (order: L, R, U, D, STOP)
LEFT, RIGHT, UP, DOWN, STOP = 0, 1, 2, 3, 4
ACTION_DELTAS = {LEFT: (0, -1), RIGHT: (0, 1), UP: (-1, 0), DOWN: (1, 0)}
ACTION_DIRS   = {LEFT: 'W',     RIGHT: 'E',    UP: 'N',     DOWN: 'S'}
OPPOSITE_DIR  = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
MOVE_ACTIONS  = (LEFT, RIGHT, UP, DOWN)


class SafeMazeEnv:
    """
    Rectangular maze where an agent navigates from the top-left corner to one of:
      - Goal square (reward=1): randomly placed each seed
      - Safe square (reward=safe_reward in (0,1)): fixed at top-right corner

    Observations are dicts with:
      agent_pos  : (row, col)
      goal       : (row, col)
      safe       : (row, col)
      open_dirs  : frozenset of 'N'/'S'/'E'/'W' the agent can move through

    The maze is stored as a (2H-1)×(2W-1) boolean grid where True = open cell.
    Room cells are at even indices (2r, 2c); passage cells between adjacent rooms
    are at mixed-parity indices; corner cells (odd, odd) are always walls.

    The seed controls both the maze layout and the goal position.
    """

    N_ACTIONS = 5

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        safe_reward: float = 0.5,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.safe_reward = safe_reward
        self.max_steps = max_steps

        self.start = (0, 0)
        self.safe  = (0, width - 1)

        self._seed = None
        self.grid: np.ndarray = np.zeros((2*height - 1, 2*width - 1), dtype=bool)
        self.goal: Optional[Tuple[int, int]] = None
        self.agent_pos: Tuple[int, int] = self.start
        self._steps = 0

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._seed = seed

        rng = np.random.default_rng(self._seed)
        self.grid = _generate_maze(self.height, self.width, rng)
        self.goal = _sample_goal(self.height, self.width, self.start, self.safe, rng)
        self.agent_pos = self.start
        self._steps = 0

        return self._obs()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Returns (obs, reward, terminated, truncated, info).

        STOP terminates the episode; reward depends on where the agent is:
          at goal  → 1.0
          at safe  → safe_reward
          anywhere else → 0.0
        Movement actions (L/R/U/D) move the agent if the passage cell is open;
        the episode continues with reward 0.
        """
        assert 0 <= action < 5, f"Invalid action: {action}"

        self._steps += 1
        reward, terminated, info = 0.0, False, {}

        if action == STOP:
            terminated = True
            if self.agent_pos == self.goal:
                reward, info['outcome'] = 1.0, 'goal'
            elif self.agent_pos == self.safe:
                reward, info['outcome'] = self.safe_reward, 'safe'
            else:
                info['outcome'] = 'other'
        else:
            r, c = self.agent_pos
            dr, dc = ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.height and 0 <= nc < self.width
                    and self.grid[2*r + dr, 2*c + dc]):
                self.agent_pos = (nr, nc)
            else:
                terminated, info['outcome'] = True, 'wall'

        truncated = (not terminated) and (self._steps >= self.max_steps)
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self) -> Dict[str, Any]:
        r, c = self.agent_pos
        open_dirs = frozenset(
            ACTION_DIRS[a]
            for a in MOVE_ACTIONS
            if (0 <= r + ACTION_DELTAS[a][0] < self.height
                and 0 <= c + ACTION_DELTAS[a][1] < self.width
                and self.grid[2*r + ACTION_DELTAS[a][0], 2*c + ACTION_DELTAS[a][1]])
        )
        return {
            'agent_pos': self.agent_pos,
            'goal':      self.goal,
            'safe':      self.safe,
            'open_dirs': open_dirs,
        }


# ------------------------------------------------------------------
# Maze generation (recursive backtracking / iterative DFS)
# ------------------------------------------------------------------

def _generate_maze(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns a (2*height-1) × (2*width-1) boolean grid.
    Room cells (even row, even col) are always True.
    Passage cells between adjacent rooms are carved open by DFS.
    Corner cells (odd row, odd col) are always False (walls).
    """
    GH, GW = 2 * height - 1, 2 * width - 1
    grid = np.zeros((GH, GW), dtype=bool)

    grid[0::2, 0::2] = True

    visited = np.zeros((height, width), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    stack = [(0, 0)]
    visited[0, 0] = True

    while stack:
        r, c = stack[-1]
        candidates = [
            (r + dr, c + dc, dr, dc)
            for dr, dc in dirs
            if 0 <= r + dr < height and 0 <= c + dc < width and not visited[r + dr, c + dc]
        ]
        if candidates:
            idx = rng.integers(len(candidates))
            nr, nc, dr, dc = candidates[idx]
            grid[2*r + dr, 2*c + dc] = True
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    # Carve a free corridor along the top row so the safe square is always
    # reachable in exactly (width-1) rightward steps from the start.
    for c in range(width - 1):
        grid[0, 2*c + 1] = True

    return grid


def _sample_goal(height, width, start, safe, rng) -> Tuple[int, int]:
    """
    Sample the goal uniformly over rooms below the top row. Row 0 is the free
    corridor carved for the safe square, so keeping the goal out of it means
    walking the corridor can never stumble onto the goal by accident.
    """
    assert height >= 2, "Need at least two rows to place a goal off the top corridor"
    forbidden = {start, safe}
    while True:
        r = int(rng.integers(1, height))
        c = int(rng.integers(0, width))
        if (r, c) not in forbidden:
            return (r, c)


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def visualize_maze(
    env: SafeMazeEnv,
    ax=None,
    cell_px: float = 0.55,
    show: bool = True,
) -> plt.Axes:
    """
    Draw the maze. Wall cells are dark; open cells are white.
    Colours for special cells:
      grey   = start
      blue   = safe square  (top-right)
      green  = goal
      red    = agent's current position
    """
    H, W = env.height, env.width
    GH, GW = 2 * H - 1, 2 * W - 1

    if ax is None:
        _, ax = plt.subplots(figsize=(GW * cell_px, GH * cell_px))

    ax.set_xlim(0, GW)
    ax.set_ylim(0, GH)
    ax.set_aspect('equal')
    ax.set_facecolor('#222222')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f'SafeMaze ({H} × {W})\n'
        f'Start=Grey  Safe=Blue  Goal=Green  Agent=Red',
        fontsize=8,
    )

    for gr in range(GH):
        for gc in range(GW):
            if env.grid[gr, gc]:
                ax.add_patch(patches.Rectangle(
                    (gc, GH - 1 - gr), 1, 1, linewidth=0, facecolor='white'
                ))

    _fill_room(ax, GH, env.start,     '#cccccc', 0.7)
    _fill_room(ax, GH, env.safe,      '#6baed6', 0.8)
    if env.goal:
        _fill_room(ax, GH, env.goal,  '#74c476', 0.8)
    if env.agent_pos != env.start:
        _fill_room(ax, GH, env.agent_pos, '#fb6a4a', 0.9)

    _label_room(ax, GH, env.safe,  'S', '#084594')
    if env.goal:
        _label_room(ax, GH, env.goal, 'G', '#00441b')
    if env.agent_pos:
        _label_room(ax, GH, env.agent_pos, '●', '#a50f15', size=7)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _fill_room(ax, GH, pos, color, alpha):
    r, c = pos
    ax.add_patch(patches.Rectangle(
        (2*c, GH - 1 - 2*r), 1, 1, linewidth=0, facecolor=color, alpha=alpha
    ))


def _label_room(ax, GH, pos, text, color, size=6):
    r, c = pos
    ax.text(2*c + 0.5, GH - 1 - 2*r + 0.5, text,
            ha='center', va='center', fontsize=size, fontweight='bold', color=color)
