"""
benchmark.py - Benchmarking harness for fair comparison of NEAT variants.

We evaluate each algorithm with K independent seeds (the algorithm's RNG seed,
not the env's). For each seed, we record the per-generation best & mean fitness
and the generation-to-solve (first gen where mean reward over a fresh evaluation
batch >= the solve threshold for N consecutive evaluations).

We also track wall-time so we can compare efficiency, and we report the final
network size to gauge parsimony.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, "/home/z/my-project")

from modern_neat.src.env_wrapper import evaluate_genome
from modern_neat.src.neat_baseline import NEATBaseline, NEATConfig


@dataclass
class RunResult:
    algorithm: str
    env_id: str
    seed: int
    history: List[dict] = field(default_factory=list)
    final_eval: dict = field(default_factory=dict)
    solved_at_gen: Optional[int] = None  # first gen where eval >= threshold
    total_wall_time_s: float = 0.0
    config: dict = field(default_factory=dict)


def evaluate_final(net_builder: Callable, env_id: str, n_eval_episodes: int = 20,
                    max_steps: int = 1000, seed_offset: int = 99999) -> dict:
    """Run a final evaluation of the best genome on a fresh batch of seeds."""
    net = net_builder()
    rewards = []
    for ep in range(n_eval_episodes):
        r, info = evaluate_genome(net, env_id, n_episodes=1,
                                    max_steps=max_steps,
                                    seed_offset=seed_offset + ep)
        rewards.append(r)
    rewards = np.array(rewards)
    return {
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "median": float(np.median(rewards)),
        "p25": float(np.percentile(rewards, 25)),
        "p75": float(np.percentile(rewards, 75)),
        "n_eval_episodes": n_eval_episodes,
        "rewards": rewards.tolist(),
    }


def run_algorithm(algo_cls, env_id: str, n_inputs: int, n_outputs: int,
                  config: NEATConfig, seed: int, n_generations: int,
                  solve_threshold: float = 475.0,
                  n_eval_episodes: int = 20,
                  algorithm_name: str = "neat") -> RunResult:
    """Run a single algorithm-instance for n_generations and return the result."""
    algo = algo_cls(env_id=env_id, n_inputs=n_inputs, n_outputs=n_outputs,
                    config=config, seed=seed)
    result = RunResult(
        algorithm=algorithm_name, env_id=env_id, seed=seed,
        config={k: str(v) for k, v in vars(config).items()},
    )
    t0 = time.time()
    solved_gen = None
    solve_count = 0
    for gen in range(n_generations):
        stats = algo.step()
        result.history.append(stats)
        # Check for solving
        if stats["max_fitness"] >= solve_threshold:
            solve_count += 1
            if solve_count >= 1 and solved_gen is None:
                solved_gen = stats["generation"]
        else:
            solve_count = 0
    result.solved_at_gen = solved_gen
    # Final evaluation
    if algo.best_genome is not None:
        result.final_eval = evaluate_final(
            lambda: algo.build_network(algo.best_genome),
            env_id, n_eval_episodes=n_eval_episodes,
            max_steps=config.max_steps,
        )
    result.total_wall_time_s = time.time() - t0
    return result


def aggregate_results(results: List[RunResult]) -> dict:
    """Aggregate K-seed runs into per-algorithm statistics."""
    final_means = [r.final_eval.get("mean", 0) for r in results if r.final_eval]
    solved_gens = [r.solved_at_gen for r in results if r.solved_at_gen is not None]
    wall_times = [r.total_wall_time_s for r in results]
    return {
        "n_runs": len(results),
        "final_eval_mean_of_means": float(np.mean(final_means)) if final_means else 0,
        "final_eval_std_of_means": float(np.std(final_means)) if final_means else 0,
        "final_eval_min": float(np.min(final_means)) if final_means else 0,
        "final_eval_max": float(np.max(final_means)) if final_means else 0,
        "solve_rate": len(solved_gens) / len(results) if results else 0,
        "solve_gen_mean": float(np.mean(solved_gens)) if solved_gens else None,
        "solve_gen_std": float(np.std(solved_gens)) if solved_gens else None,
        "wall_time_mean_s": float(np.mean(wall_times)) if wall_times else 0,
        "wall_time_std_s": float(np.std(wall_times)) if wall_times else 0,
    }


def save_result(result: RunResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "algorithm": result.algorithm,
            "env_id": result.env_id,
            "seed": result.seed,
            "history": result.history,
            "final_eval": result.final_eval,
            "solved_at_gen": result.solved_at_gen,
            "total_wall_time_s": result.total_wall_time_s,
            "config": result.config,
        }, f, indent=2)
