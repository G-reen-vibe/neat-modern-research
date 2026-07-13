"""
Phase 0 baseline: run canonical NEAT on multiple gym envs.

MountainCar-v0 is the canonical hard test for NEAT - sparse reward, requires
building up momentum. Acrobot-v1 is also hard. CartPole-v1 is the easy baseline.

For each env, we run K seeds and report aggregated stats.
"""
import sys
import os
sys.path.insert(0, "/home/z/my-project")
sys.path.insert(0, "/home/z/my-project/modern_neat")

import json
import time
import argparse

from modern_neat.src.neat_baseline import NEATConfig, NEATBaseline
from scripts.benchmark import run_algorithm, save_result, aggregate_results


# Per-env solve thresholds based on official gymnasium criteria
ENV_CONFIG = {
    "CartPole-v1": {"solve": 475.0, "max_steps": 500, "n_eval": 20, "n_eps": 3},
    "MountainCar-v0": {"solve": -110.0, "max_steps": 200, "n_eval": 20, "n_eps": 5},
    "Acrobot-v1": {"solve": -100.0, "max_steps": 500, "n_eval": 20, "n_eps": 3},
}

ENV_SHAPES = {
    "CartPole-v1": (4, 2),
    "MountainCar-v0": (2, 3),
    "Acrobot-v1": (6, 3),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--envs", type=str, default="CartPole-v1,MountainCar-v0,Acrobot-v1")
    parser.add_argument("--out-dir", type=str,
                        default="/home/z/my-project/modern_neat/results/phase0_baseline")
    args = parser.parse_args()

    envs = args.envs.split(",")
    os.makedirs(args.out_dir, exist_ok=True)

    all_summary = {}
    for env_id in envs:
        n_inputs, n_outputs = ENV_SHAPES[env_id]
        env_cfg = ENV_CONFIG[env_id]
        cfg = NEATConfig(
            pop_size=args.pop_size,
            n_episodes=env_cfg["n_eps"],
            max_steps=env_cfg["max_steps"],
            add_node_rate=0.03,
            add_conn_rate=0.05,
            weight_mut_rate=0.8,
            bias_mut_rate=0.8,
            compat_threshold=3.0,
            target_species=10,
        )
        env_dir = os.path.join(args.out_dir, env_id)
        os.makedirs(env_dir, exist_ok=True)
        results = []
        print(f"\n========== {env_id} ==========")
        for s in range(args.seeds):
            print(f"=== {env_id} seed {s} ===")
            result = run_algorithm(
                NEATBaseline, env_id=env_id, n_inputs=n_inputs, n_outputs=n_outputs,
                config=cfg, seed=s, n_generations=args.generations,
                solve_threshold=env_cfg["solve"], n_eval_episodes=env_cfg["n_eval"],
                algorithm_name="vanilla_neat",
            )
            save_result(result, os.path.join(env_dir, f"seed_{s}.json"))
            results.append(result)
            print(f"  final_eval mean={result.final_eval.get('mean',0):.1f} "
                  f"solved_at_gen={result.solved_at_gen} "
                  f"time={result.total_wall_time_s:.1f}s")
        agg = aggregate_results(results)
        with open(os.path.join(env_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        all_summary[env_id] = agg
        print(f"  AGG: solve_rate={agg['solve_rate']:.2f} "
              f"final_mean={agg['final_eval_mean_of_means']:.1f} "
              f"wall_time={agg['wall_time_mean_s']:.1f}s")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
