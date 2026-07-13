"""
env_wrapper.py - Lightweight wrappers around gymnasium envs with caching.

For RL benchmarks we evaluate each genome by running N episodes and taking the
mean (or min, configurable) return. We seed envs deterministically per-evaluation
so the comparison between genomes is fair.

Optimization: We cache a single env per (env_id) and just call reset() with the
new seed between episodes - much faster than gym.make() each time.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple

from .network import Network


_ENV_CACHE: dict = {}


def get_env(env_id: str):
    """Cache envs to avoid the cost of gym.make() on every call."""
    if env_id not in _ENV_CACHE:
        _ENV_CACHE[env_id] = gym.make(env_id)
    return _ENV_CACHE[env_id]


def evaluate_genome(net: Network, env_id: str, n_episodes: int = 1,
                    max_steps: int = 1000, seed_offset: int = 0,
                    render: bool = False) -> Tuple[float, dict]:
    """Run `n_episodes` rollouts of `net` on `env_id` and return mean return.

    Uses a cached env to avoid gym.make overhead."""
    env = get_env(env_id)
    total = 0.0
    steps_total = 0
    rewards = []
    is_discrete = env.action_space.__class__.__name__ == "Discrete"
    action_low = getattr(env.action_space, "low", None)
    action_high = getattr(env.action_space, "high", None)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_reward = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            out = net.forward(obs)
            if is_discrete:
                action = int(np.argmax(out))
            else:
                action = np.clip(out, action_low, action_high)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += float(r)
            steps += 1
        rewards.append(ep_reward)
        steps_total += steps
        total += ep_reward
    info = {
        "mean_reward": total / n_episodes,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_steps": steps_total / n_episodes,
        "rewards": rewards,
    }
    return total / n_episodes, info


def collect_behavior_descriptor(net: Network, env_id: str,
                                  n_episodes: int = 1,
                                  max_steps: int = 200,
                                  seed_offset: int = 0,
                                  n_bins: int = 6) -> Tuple[float, np.ndarray, dict]:
    """Run rollouts and collect a low-dimensional behavioral descriptor.

    For CartPole, we summarize the trajectory by binning the pole angle over time.
    This is used by novelty search variants."""
    env = get_env(env_id)
    angle_bins = np.linspace(-0.21, 0.21, n_bins + 1)
    pos_bins = np.linspace(-2.4, 2.4, n_bins + 1)
    angle_hist = np.zeros(n_bins, dtype=np.float64)
    pos_hist = np.zeros(n_bins, dtype=np.float64)
    total_reward = 0.0
    counts = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < max_steps:
            out = net.forward(obs)
            action = int(np.argmax(out))
            obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            steps += 1
            angle_idx = int(np.digitize(obs[2], angle_bins) - 1)
            angle_idx = max(0, min(n_bins - 1, angle_idx))
            pos_idx = int(np.digitize(obs[0], pos_bins) - 1)
            pos_idx = max(0, min(n_bins - 1, pos_idx))
            angle_hist[angle_idx] += 1
            pos_hist[pos_idx] += 1
            counts += 1
    if counts > 0:
        angle_hist /= counts
        pos_hist /= counts
    desc = np.concatenate([angle_hist, pos_hist])
    info = {"mean_reward": total_reward / n_episodes}
    return total_reward / n_episodes, desc, info
