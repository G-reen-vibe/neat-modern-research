"""
Phase 0 baseline: run canonical NEAT on CartPole-v1 for K seeds.

Goal: establish a real baseline number for vanilla NEAT on CartPole-v1 with
multiple evaluation episodes (so we don't get fooled by lucky genomes).
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop-size", type=int, default=150)
    parser.add_argument("--n-episodes", type=int, default=3,
                        help="Episodes per genome evaluation")
    parser.add_argument("--out-dir", type=str,
                        default="/home/z/my-project/modern_neat/results/phase0_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = NEATConfig(
        pop_size=args.pop_size,
        n_episodes=args.n_episodes,
        max_steps=500,
        add_node_rate=0.03,
        add_conn_rate=0.05,
        weight_mut_rate=0.8,
        bias_mut_rate=0.8,
        compat_threshold=3.0,
        target_species=10,
    )

    all_results = []
    for s in range(args.seeds):
        print(f"\n=== seed {s} ===")
        result = run_algorithm(
            NEATBaseline, env_id="CartPole-v1", n_inputs=4, n_outputs=2,
            config=cfg, seed=s, n_generations=args.generations,
            solve_threshold=475.0, n_eval_episodes=20,
            algorithm_name="vanilla_neat",
        )
        save_result(result, os.path.join(args.out_dir, f"seed_{s}.json"))
        all_results.append(result)
        print(f"  final_eval mean={result.final_eval.get('mean',0):.1f} "
              f"solved_at_gen={result.solved_at_gen} "
              f"time={result.total_wall_time_s:.1f}s")

    agg = aggregate_results(all_results)
    with open(os.path.join(args.out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
