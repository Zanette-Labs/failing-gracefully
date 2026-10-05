"""
Training for the SafeMaze environment.

Design:
  - SafeMazeEnv stays on CPU.
  - Policy lives on GPU.
  - During rollout collection:
      * CPU envs produce observations.
      * Observations from all active envs are batched.
      * One GPU forward pass samples actions for all active envs.
      * CPU envs are stepped with those sampled actions.
  - During training:
      * Stored obs/action trajectories are forwarded again on GPU with grad.
"""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import wandb

from safe_maze.env import SafeMazeEnv
from safe_maze.config import H, W, n_blocks
from safe_maze.cnn_policy import MazeResNetPolicy, obs_to_vec


# ── hyperparameters ───────────────────────────────────────────────────────────

N_ENVS          = 32
N_ROLLOUTS      = 16
N_STEPS         = 10_000
LR              = 3e-5
MAX_GRAD_NORM   = 1.0
SAFE_REWARD     = 0.5
PRETRAIN_PATH   = str(Path(__file__).with_name("pretrained_policy.pt"))
CKPT_PATH       = None  # auto: {run_dir}/policy.pt
LOG_INTERVAL    = 10

LOGP_BATCH_SIZE = 65_536
DEFAULT_LOG_ROOT = Path("/tmp/safe_maze")


# ── advantages ────────────────────────────────────────────────────────────────

def rloo_advantages(
    rewards: torch.Tensor,
    n_envs: int,
    n_rollouts: int,
    **_,
) -> torch.Tensor:
    """
    A_i = r_i - mean(r_{j != i})
    """
    r = rewards.view(n_envs, n_rollouts)
    group_sum = r.sum(dim=1, keepdim=True)
    baselines = (group_sum - r) / (n_rollouts - 1)
    return (r - baselines).view(-1)


def grpo_advantages(
    rewards: torch.Tensor,
    n_envs: int,
    n_rollouts: int,
    eps: float = 1e-8,
    **_,
) -> torch.Tensor:
    """
    A_i = (r_i - mean(r)) / std(r)
    """
    r = rewards.view(n_envs, n_rollouts)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    return ((r - mean) / (std + eps)).view(-1)


def gracefulrl_advantages(
    rewards: torch.Tensor,
    n_envs: int,
    n_rollouts: int,
    eps: float = 0.0,
    **_,
) -> torch.Tensor:
    """
    GracefulRL advantage: sort rewards descending, then
    a_(i) = sum_{j=i}^{N} [R(j) - R(j+1)] / (j + eps)   (R(N+1) = 0),
    centered by group mean. `eps` smooths the denominator, damping the
    large weight the leading terms otherwise receive.
    """
    r = rewards.view(n_envs, n_rollouts)
    device, dtype = r.device, r.dtype
    n = n_rollouts

    r_sorted, order = torch.sort(r, dim=1, descending=True)
    r_ext = torch.cat([r_sorted, torch.zeros(n_envs, 1, device=device, dtype=dtype)], dim=1)

    deltas = r_ext[:, :-1] - r_ext[:, 1:]
    denom = torch.arange(1, n + 1, device=device, dtype=dtype).unsqueeze(0) + eps
    w = deltas / denom

    a_sorted = torch.flip(torch.cumsum(torch.flip(w, dims=[1]), dim=1), dims=[1])

    inv = torch.empty_like(order)
    inv.scatter_(1, order, torch.arange(n, device=device, dtype=order.dtype).unsqueeze(0).expand(n_envs, -1))
    a = a_sorted.gather(1, inv)

    a = a - a.mean(dim=1, keepdim=True)
    return a.view(-1)


ADVANTAGE_FNS = {
    "rloo": rloo_advantages,
    "grpo": grpo_advantages,
    "gracefulrl": gracefulrl_advantages,
}


class SafeRewardSampler:
    def __init__(
        self,
        mode: str,
        constant: float = SAFE_REWARD,
        online_multiplier: float = 1.0,
        online_ema_decay: float = 0.9,
    ):
        if mode not in ("constant", "online_scaled"):
            raise ValueError(f"Unknown safe_reward_mode: {mode}")
        if not 0 <= online_multiplier <= 1:
            raise ValueError("online_multiplier must be between 0 and 1")
        if not math.isfinite(online_ema_decay) or not 0 <= online_ema_decay < 1:
            raise ValueError("online_ema_decay must be finite and between 0 and 1 (exclusive)")
        self.mode = mode
        self.constant = constant
        self._ema = 0
        self.online_multiplier = online_multiplier
        self.online_ema_decay = online_ema_decay

    def update(self, frac_goal: float) -> None:
        if not 0 <= frac_goal <= 1:
            raise ValueError("frac_goal must be between 0 and 1")
        if self.mode == "online_scaled":
            self._ema = self.online_ema_decay * self._ema + (1 - self.online_ema_decay) * frac_goal

    def sample(self, n: int) -> np.ndarray:
        """Return constant rewards or rewards based on the previous goal-rate EMA."""
        value = (
            self.constant
            if self.mode == "constant"
            else self.online_multiplier * self._ema
        )
        return np.full(n, value, dtype=np.float32)


def pg_loss(
    advantages: torch.Tensor,
    traj_logp: torch.Tensor,
    **_,
) -> torch.Tensor:
    """
    L = mean_i[-A_i * log p_i]
    """
    return -(advantages.detach() * traj_logp).mean()


def _expand_step_advantages(
    advantages: torch.Tensor,
    lengths: list[int],
    n_steps: int,
) -> torch.Tensor:
    lengths_t = torch.as_tensor(
        lengths, device=advantages.device, dtype=torch.long
    )
    if lengths_t.numel() != advantages.numel():
        raise ValueError(
            f"Expected one trajectory length per advantage, got "
            f"{lengths_t.numel()} lengths and {advantages.numel()} advantages"
        )
    if lengths_t.sum().item() != n_steps:
        raise ValueError(
            f"Trajectory lengths sum to {lengths_t.sum().item()}, "
            f"but there are {n_steps} step log-probabilities"
        )
    return torch.repeat_interleave(advantages.detach(), lengths_t)


def dpg_loss(
    advantages: torch.Tensor,
    step_logp: torch.Tensor,
    lengths: list[int],
    dpg_eta: float = 0.9,
    **_,
) -> torch.Tensor:
    """
    Delightful policy gradient: https://arxiv.org/abs/2603.14608
    L = -1/N sum_i sum_t sigmoid(A_i * -log p_it / eta) * A_i * log p_it
    """
    if not math.isfinite(dpg_eta) or dpg_eta <= 0:
        raise ValueError(f"dpg_eta must be finite and positive, got {dpg_eta}")

    step_advantages = _expand_step_advantages(
        advantages, lengths, n_steps=step_logp.numel()
    )

    # DPG gates each sampled action using that action's surprisal. Gating once
    # with summed trajectory surprisal makes eta depend on episode length and
    # can saturate the gate merely because a trajectory contains more steps.
    surprisal = -step_logp.detach()
    delight = step_advantages * surprisal
    gate = torch.sigmoid(delight / dpg_eta)
    return -((gate * step_advantages) * step_logp).sum() / advantages.numel()


LOSS_FNS = {
    "dpg": dpg_loss,
    "pg": pg_loss,
}


def _comb_ratio(n: int, c_missing: int, k: int) -> float:
    """C(n - c_missing, k) / C(n, k): prob of zero successes in k draws w/o replacement."""
    if c_missing == 0:
        return 1.0
    available = n - c_missing
    if available < k:
        return 0.0
    result = 1.0
    for i in range(k):
        result *= (available - i) / (n - i)
    return result


def estimate_best_at_k(
    rewards: torch.Tensor,
    n_envs: int,
    n_rollouts: int,
    safe_rewards: np.ndarray,
) -> dict:
    """
    Unbiased estimator of E[best reward in k rollouts] for k = 1, 2, 4, ..., n_rollouts.

    For each env, count c1 (goal), cs (safe), c0 (zero) out of n_rollouts.
    E[best@k] = (1 - P_no_goal_k) + safe_reward * (P_no_goal_k - P_no_goal_no_safe_k)
    where P_... uses the hypergeometric C(n-c, k)/C(n, k) ratio.
    """
    r_mat = rewards.view(n_envs, n_rollouts).tolist()
    result = {}
    k = 1
    while True:
        total = 0.0
        for env_idx, row in enumerate(r_mat):
            sr = float(safe_rewards[env_idx])
            c1 = sum(1 for r in row if r == 1.0)
            # use math.isclose because equality is not possible after GPU tensor ops
            cs = sum(1 for r in row if math.isclose(r, sr, abs_tol=1e-6) and r != 1.0)
            p_no_goal           = _comb_ratio(n_rollouts, c1,      k)
            p_no_goal_no_safe   = _comb_ratio(n_rollouts, c1 + cs, k)
            total += (1 - p_no_goal) + sr * (p_no_goal - p_no_goal_no_safe)
        result[f"best@{k}"] = total / n_envs
        if k >= n_rollouts:
            break
        k = min(k * 2, n_rollouts)
    return result


# ── GPU-batched rollout collection ────────────────────────────────────────────

@torch.no_grad()
def collect_rollouts_gpu_policy_cpu_env(
    policy: torch.nn.Module,
    seeds: np.ndarray,
    n_rollouts: int,
    safe_rewards: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
):
    """
    Collect rollouts with CPU envs and GPU policy forward passes.

    We create n_envs * n_rollouts CPU env instances.

    Group structure is preserved:
        trajs[env_idx * n_rollouts + rollout_idx]

    deterministic=True selects argmax actions (ties choose the first index).
    The default retains stochastic sampling for training.

    Returns:
        trajs: list of (obs_arr, act_arr, reward)
            obs_arr: float32 array, shape [T, *obs_shape]
            act_arr: int64 array, shape [T]
            reward: float
    """
    policy.eval()

    env_specs = []
    for seed_idx, seed in enumerate(seeds):
        for rollout_idx in range(n_rollouts):
            env_specs.append((int(seed), seed_idx, rollout_idx))

    envs = [
        SafeMazeEnv(
            width=W,
            height=H,
            seed=seed,
            safe_reward=float(safe_rewards[seed_idx]),
        )
        for seed, seed_idx, _ in env_specs
    ]

    for env in envs:
        env.reset()

    batch_size = len(envs)

    active = np.ones(batch_size, dtype=bool)
    obs_buf    = [[] for _ in range(batch_size)]
    act_buf    = [[] for _ in range(batch_size)]
    rew_buf    = [None for _ in range(batch_size)]
    status_buf = [None for _ in range(batch_size)]

    while active.any():
        active_idxs = np.flatnonzero(active)

        # CPU observation construction.
        obs_np = np.stack([
            obs_to_vec(envs[i]) for i in active_idxs
        ]).astype(np.float32)

        # GPU policy forward.
        obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True)
        logits = policy(obs_t)

        actions_t = (logits.argmax(dim=-1) if deterministic
                     else torch.distributions.Categorical(logits=logits).sample())
        actions = actions_t.detach().cpu().numpy()

        # CPU env stepping.
        for local_j, env_idx in enumerate(active_idxs):
            action = int(actions[local_j])
            obs_before_action = obs_np[local_j]

            obs_buf[env_idx].append(obs_before_action)
            act_buf[env_idx].append(action)

            _, reward, terminated, truncated, info = envs[env_idx].step(action)

            if terminated or truncated:
                if truncated:
                    # A timeout does not sample another action. In particular,
                    # do not assign a policy gradient to an invented STOP.
                    status_buf[env_idx] = 'timeout'
                else:
                    status_buf[env_idx] = info.get('outcome', 'unknown')

                rew_buf[env_idx] = float(reward)
                active[env_idx] = False

    trajs = []
    for i in range(batch_size):
        trajs.append((
            np.stack(obs_buf[i]).astype(np.float32),
            np.asarray(act_buf[i], dtype=np.int64),
            float(rew_buf[i]),
            status_buf[i],
        ))

    return trajs


def compute_step_logps_gpu(
    policy: torch.nn.Module,
    obs_np: np.ndarray,
    acts_np: np.ndarray,
    device: torch.device,
    batch_size: int = LOGP_BATCH_SIZE,
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Recompute log-probs with grad on GPU.

    This is separated from rollout collection because rollout action sampling
    happens under torch.no_grad(), but the policy-gradient loss needs gradients.
    """
    policy.train()

    logps = []
    entropies = []

    n = len(acts_np)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        obs_t = torch.from_numpy(obs_np[start:end]).to(device, non_blocking=True)
        acts_t = torch.from_numpy(acts_np[start:end]).to(device, non_blocking=True)

        log_probs = torch.log_softmax(policy(obs_t), dim=-1)

        step_logp = log_probs.gather(1, acts_t.unsqueeze(1)).squeeze(1)
        logps.append(step_logp)
        if return_entropy:
            entropies.append(-(log_probs.exp() * log_probs).sum(dim=-1))

    step_logps = torch.cat(logps, dim=0)
    if return_entropy:
        return step_logps, torch.cat(entropies, dim=0)
    return step_logps


# ── training loop ─────────────────────────────────────────────────────────────


def backward_in_batches(
    policy: torch.nn.Module,
    obs_np: np.ndarray,
    acts_np: np.ndarray,
    lengths: list[int],
    advantages: torch.Tensor,
    device: torch.device,
    entropy_coef: float,
    batch_size: int,
    loss: str = "pg",
    dpg_eta: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backpropagate the full-batch objective while freeing each chunk's graph.

    Normalize policy terms by the total trajectory count and entropy by the
    total action count. The optimizer steps only after all chunks contribute.
    Returns total loss, policy loss, and mean entropy.
    """
    if batch_size < 1:
        raise ValueError("gradient_batch_size must be positive")
    if loss not in LOSS_FNS:
        raise ValueError(f"Unknown loss scheme: {loss}")
    if loss == "dpg" and (not math.isfinite(dpg_eta) or dpg_eta <= 0):
        raise ValueError("dpg_eta must be finite and positive")
    n_actions = len(acts_np)
    n_trajs = len(lengths)
    step_advantages = _expand_step_advantages(advantages, lengths, n_actions)
    policy_total = torch.zeros((), device=device)
    entropy_total = torch.zeros((), device=device)
    for start in range(0, n_actions, batch_size):
        end = min(start + batch_size, n_actions)
        logps, entropies = compute_step_logps_gpu(
            policy, obs_np[start:end], acts_np[start:end], device,
            batch_size=batch_size, return_entropy=True,
        )
        weights = step_advantages[start:end]
        if loss == "dpg":
            gate = torch.sigmoid(weights * -logps.detach() / dpg_eta)
            weights = gate * weights
        policy_part = -(weights * logps).sum() / n_trajs
        entropy_part = entropies.sum() / n_actions
        loss_part = policy_part - entropy_coef * entropy_part
        if not torch.isfinite(loss_part):
            raise FloatingPointError("Non-finite loss in gradient batch")
        loss_part.backward()
        policy_total += policy_part.detach()
        entropy_total += entropy_part.detach()
    return policy_total - entropy_coef * entropy_total, policy_total, entropy_total


def train(
    n_steps: int = N_STEPS,
    n_envs: int = N_ENVS,
    n_rollouts: int = N_ROLLOUTS,
    lr: float = LR,
    pretrain_path: str = PRETRAIN_PATH,
    ckpt_path: str | None = CKPT_PATH,
    advantage: str = "rloo",
    from_scratch: bool = False,
    device: str | None = None,
    safe_reward: float = SAFE_REWARD,
    safe_reward_mode: str = "constant",
    gracefulrl_eps: float = 0.0,
    save_ckpt: bool = True,
    entropy_coef: float = 0.0,
    online_multiplier: float = 1.0,
    online_ema_decay: float = 0.9,
    seed: int = 0,
    wandb_project: str = "maze_rollout_sweep_final",
    wandb_group: str | None = None,
    run_name: str | None = None,
    run_dir: str | None = None,
    gradient_batch_size: int | None = None,
    checkpoint_interval: int = 0,
    loss: str = "pg",
    dpg_eta: float = 0.9,
) -> MazeResNetPolicy:

    assert n_rollouts >= 2, "Advantage estimation requires at least 2 rollouts per group"
    assert advantage in ADVANTAGE_FNS, f"Unknown advantage scheme: {advantage}"
    if loss not in LOSS_FNS:
        raise ValueError(f"Unknown loss scheme: {loss}")
    if loss == "dpg" and (not math.isfinite(dpg_eta) or dpg_eta <= 0):
        raise ValueError("dpg_eta must be finite and positive")
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("entropy_coef must be finite and nonnegative")
    if n_steps < 1 or n_envs < 1:
        raise ValueError("n_steps and n_envs must be positive")
    if gradient_batch_size is not None and gradient_batch_size < 1:
        raise ValueError("gradient_batch_size must be positive")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be nonnegative")
    if run_dir is None:
        DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        run_dir = tempfile.mkdtemp(prefix="run-", dir=DEFAULT_LOG_ROOT)
    output_dir = Path(run_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    if ckpt_path is None:
        ckpt_path = str(output_dir / "policy.pt")
    if save_ckpt:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    # Keep W&B's auxiliary writes off the home-directory quota as well.
    os.environ["WANDB_DIR"] = str(output_dir)
    for variable, subdir in (
        ("WANDB_DATA_DIR", "wandb-data"),
        ("WANDB_CACHE_DIR", "wandb-cache"),
        ("WANDB_CONFIG_DIR", "wandb-config"),
    ):
        directory = output_dir / subdir
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)

    adv_fn = ADVANTAGE_FNS[advantage]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    policy = MazeResNetPolicy(n_blocks=n_blocks)
    if not from_scratch:
        policy.load_state_dict(
            torch.load(pretrain_path, map_location="cpu", weights_only=True)
        )
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    sampler = SafeRewardSampler(
        mode=safe_reward_mode,
        constant=safe_reward,
        online_multiplier=online_multiplier,
        online_ema_decay=online_ema_decay,
    )

    sr_tag = (
        str(safe_reward)
        if safe_reward_mode == "constant"
        else f"online_ema{online_ema_decay}_x{online_multiplier}"
    )
    default_run_name = f"{advantage}_sr{sr_tag}_rollouts{n_rollouts}_lr{lr}_ent{entropy_coef}_seed{seed}"
    if advantage == "gracefulrl":
        default_run_name += f"_eps{gracefulrl_eps}"
    if loss == "dpg":
        default_run_name = f"dpg_eta{dpg_eta}_{default_run_name}"

    run = wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=run_name or default_run_name,
        dir=str(output_dir),
        config=dict(
            n_steps=n_steps,
            n_envs=n_envs,
            n_rollouts=n_rollouts,
            lr=lr,
            safe_reward=safe_reward,
            safe_reward_mode=safe_reward_mode,
            max_grad_norm=MAX_GRAD_NORM,
            H=H,
            W=W,
            advantage=advantage,
            loss=loss,
            dpg_eta=dpg_eta,
            gracefulrl_eps=gracefulrl_eps,
            save_ckpt=save_ckpt,
            device=str(device),
            entropy_coef=entropy_coef,
            entropy_reduction="mean_over_sampled_actions",
            online_multiplier=online_multiplier,
            online_ema_decay=online_ema_decay,
            online_pass_rate_definition="ema_of_previous_batch_goal_fractions_initially_zero",
            seed=seed,
            pretrain_path=pretrain_path,
            from_scratch=from_scratch,
            n_blocks=n_blocks,
            timeout_appends_stop=False,
            gradient_batch_size=gradient_batch_size,
            checkpoint_interval=checkpoint_interval,
            run_dir=str(output_dir),
            ckpt_path=ckpt_path,
        ),
    )
    if output_dir is not None:
        (output_dir / "config.json").write_text(json.dumps(dict(run.config), indent=2) + "\n")
        (output_dir / "wandb_url.txt").write_text((run.url or "offline") + "\n")
    metrics_file = (output_dir / "metrics.jsonl").open("x", buffering=1) if output_dir else None
    recent_goal = deque(maxlen=200)
    recent_safe = deque(maxlen=200)
    recent_entropy = deque(maxlen=200)
    started_at = time.monotonic()
    completed = False

    print(
        f"RL training: {n_steps} steps | "
        f"{n_envs} envs × {n_rollouts} rollouts | "
        f"device={device} | advantage={advantage} | loss={loss} | dpg_eta={dpg_eta} | entropy_coef={entropy_coef} | "
        f"safe_reward_mode={safe_reward_mode} | online_multiplier={online_multiplier} | "
        f"online_ema_decay={online_ema_decay} | seed={seed}"
    )
    print(f"Run directory: {output_dir}")
    if from_scratch:
        print("Training from scratch (random init)\n")
    else:
        print(f"Loaded pretrained weights from {pretrain_path}\n")
    print(
        f'{"step":>5}  {"loss":>8}  {"avg_r":>6}  '
        f'{"goal":>5}  {"safe":>5}  {"zero":>5}  {"avg_len":>7}  {"ema":>6}'
    )
    print("-" * 64)

    try:
        for step in range(1, n_steps + 1):
            step_started_at = time.monotonic()

            # ── 1. collect CPU-env rollouts with GPU policy calls ──────────────
            seeds = rng.integers(0, 1_000_000, size=n_envs)
            safe_rewards = sampler.sample(n_envs)
            reward_pass_rate = sampler._ema
            trajs = collect_rollouts_gpu_policy_cpu_env(
                policy=policy,
                seeds=seeds,
                n_rollouts=n_rollouts,
                safe_rewards=safe_rewards,
                device=device,
            )

            # trajs[env_idx * n_rollouts + rollout_idx] preserves group structure.

            # ── 2. prepare trajectory tensors ─────────────────────────────────
            lengths = [len(t[1]) for t in trajs]

            rewards = torch.tensor(
                [t[2] for t in trajs],
                dtype=torch.float32,
                device=device,
            )

            obs_np = np.concatenate([t[0] for t in trajs], axis=0).astype(np.float32)
            acts_np = np.concatenate([t[1] for t in trajs], axis=0).astype(np.int64)

            # ── 3. advantages + gradient calculation ─────────────────────────
            advantages = adv_fn(
                rewards=rewards,
                n_envs=n_envs,
                n_rollouts=n_rollouts,
                **({"eps": gracefulrl_eps} if advantage == "gracefulrl" else {}),
            )

            optimizer.zero_grad(set_to_none=True)
            if gradient_batch_size is not None:
                train_loss, policy_loss, entropy = backward_in_batches(
                    policy, obs_np, acts_np, lengths, advantages, device,
                    entropy_coef, gradient_batch_size, loss=loss, dpg_eta=dpg_eta,
                )
            else:
                step_logp, step_entropy = compute_step_logps_gpu(
                    policy=policy, obs_np=obs_np, acts_np=acts_np, device=device,
                    batch_size=LOGP_BATCH_SIZE, return_entropy=True,
                )
                traj_logps = []
                offset = 0
                for length in lengths:
                    traj_logps.append(step_logp[offset: offset + length].sum())
                    offset += length
                traj_logp = torch.stack(traj_logps, dim=0)
                policy_loss = LOSS_FNS[loss](
                    advantages, traj_logp=traj_logp, step_logp=step_logp,
                    lengths=lengths, dpg_eta=dpg_eta,
                )
                entropy = step_entropy.mean()
                train_loss = policy_loss - entropy_coef * entropy
                if not torch.isfinite(train_loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                train_loss.backward()

            # ── 4. one optimizer update for the complete rollout batch ───────
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM,
                                                       error_if_nonfinite=True)
            optimizer.step()

            # ── 5. logging ────────────────────────────────────────────────────
            with torch.no_grad():
                statuses  = [t[3] for t in trajs]
                n_trajs   = len(statuses)
                avg_r     = rewards.mean().item()
                avg_len   = float(np.mean(lengths))
                frac_goal = sum(s == 'goal'              for s in statuses) / n_trajs
                frac_safe = sum(s == 'safe'              for s in statuses) / n_trajs
                frac_zero = sum(s not in ('goal', 'safe') for s in statuses) / n_trajs
                sampler.update(frac_goal)

                best_at_k = estimate_best_at_k(rewards, n_envs, n_rollouts, safe_rewards)

            recent_goal.append(frac_goal)
            recent_safe.append(frac_safe)
            recent_entropy.append(entropy.item())
            metrics = {
                "train/loss":            train_loss.item(),
                "train/policy_loss":     policy_loss.item(),
                "train/entropy":         entropy.item(),
                "train/entropy_bonus":   entropy_coef * entropy.item(),
                "train/grad_norm":       grad_norm.item(),
                "train/avg_r":           avg_r,
                "train/avg_len":         avg_len,
                "train/frac_goal":       frac_goal,
                "train/frac_safe":       frac_safe,
                "train/frac_zero":       frac_zero,
                "train/goal_rate_last200": float(np.mean(recent_goal)),
                "train/safe_rate_last200": float(np.mean(recent_safe)),
                "train/reward_pass_rate": reward_pass_rate,
                "train/online_pass_rate": frac_goal,
                "train/avg_safe_reward": float(safe_rewards.mean()),
                "train/ema_frac_goal":   sampler._ema,
                "timing/step_seconds": time.monotonic() - step_started_at,
                "timing/elapsed_seconds": time.monotonic() - started_at,
                **{f"best_metrics/{k}": v for k, v in best_at_k.items()},
            }
            # Do not emit NaN placeholders for reward groups absent from a batch.
            metrics = {key: value for key, value in metrics.items() if math.isfinite(value)}
            wandb.log(metrics, step=step)
            if metrics_file is not None:
                metrics_file.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")

            if step % LOG_INTERVAL == 0 or step == 1:
                print(
                    f"{step:>5}  {train_loss.item():>8.4f}  {avg_r:>6.3f}  "
                    f"{frac_goal:>5.2f}  {frac_safe:>5.2f}  "
                    f"{frac_zero:>5.2f}  {avg_len:>7.1f}  {sampler._ema:>6.4f}"
                )

            if save_ckpt and checkpoint_interval and step % checkpoint_interval == 0 and step < n_steps:
                periodic_path = Path(ckpt_path).with_name(f"{Path(ckpt_path).stem}_step_{step:05d}.pt")
                torch.save(policy.state_dict(), periodic_path)
                print(f"Saved intermediate checkpoint → {periodic_path}")

        if save_ckpt:
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Saved → {ckpt_path}")
        summary = {
            "status": "completed",
            "completed_steps": n_steps,
            "final200/goal_rate": float(np.mean(recent_goal)),
            "final200/safe_rate": float(np.mean(recent_safe)),
            "final200/entropy": float(np.mean(recent_entropy)),
            "elapsed_seconds": time.monotonic() - started_at,
        }
        run.summary.update(summary)
        if output_dir is not None:
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        completed = True

    finally:
        if metrics_file is not None:
            metrics_file.close()
        if not completed:
            run.summary["status"] = "failed_or_interrupted"
        wandb.finish(exit_code=0 if completed else 1)

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--advantage", choices=list(ADVANTAGE_FNS), default="rloo")
    parser.add_argument("--loss", choices=list(LOSS_FNS), default="pg",
                        help="Policy-gradient loss; DPG applies a detached per-action delight gate.")
    parser.add_argument("--dpg_eta", type=float, default=0.9,
                        help="Positive finite temperature for the DPG sigmoid gate.")
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--n_envs", type=int, default=N_ENVS)
    parser.add_argument("--n_rollouts", type=int, default=N_ROLLOUTS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--safe_reward", type=float, default=SAFE_REWARD)
    parser.add_argument("--safe_reward_mode", choices=["constant", "online_scaled"], default="constant")
    parser.add_argument("--checkpoint_interval", type=int, default=0)
    parser.add_argument("--online_multiplier", type=float, default=1.0)
    parser.add_argument("--online_ema_decay", type=float, default=0.9,
                        help="EMA weight on the previous goal-rate estimate.")
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", default="maze_rollout_sweep_final")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_dir", default=None,
                        help="Directory for metrics and W&B files; defaults to a unique /tmp/safe_maze/run-* directory.")
    parser.add_argument("--gradient_batch_size", type=int, default=None,
                        help="Limit actions per gradient chunk; preserves full-batch loss normalization.")
    parser.add_argument("--gracefulrl_eps", type=float, default=0.0)
    parser.add_argument(
        "--no_save_ckpt",
        action="store_false",
        dest="save_ckpt",
        help="Disable saving the final policy checkpoint.",
    )
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--pretrain_path", default=PRETRAIN_PATH)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(**vars(args))
