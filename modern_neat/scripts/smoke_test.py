"""Quick smoke test: run baseline NEAT on CartPole-v1 for 5 generations."""
import sys
sys.path.insert(0, "/home/z/my-project")

import time
from modern_neat.src.neat_baseline import NEATBaseline, NEATConfig


def main():
    cfg = NEATConfig(
        pop_size=50,
        n_episodes=1,
        max_steps=500,
        add_node_rate=0.03,
        add_conn_rate=0.05,
    )
    neat = NEATBaseline(env_id="CartPole-v1", n_inputs=4, n_outputs=2,
                        config=cfg, seed=42)
    print(f"Initial population: {len(neat.population)} genomes")
    print(f"Initial genome nodes: {neat.population[0].num_nodes}, "
          f"conns: {neat.population[0].num_conns}")

    n_gens = 5
    for i in range(n_gens):
        t0 = time.time()
        stats = neat.step()
        print(f"Gen {stats['generation']:3d} | "
              f"max={stats['max_fitness']:7.2f} | "
              f"mean={stats['mean_fitness']:7.2f} | "
              f"best_ever={stats['best_overall']:7.2f} | "
              f"species={stats['n_species']:2d} | "
              f"mean_nodes={stats['mean_nodes']:.1f} | "
              f"max_nodes={stats['max_nodes']:2d} | "
              f"mean_conns={stats['mean_conns']:.1f} | "
              f"time={stats['wall_time_s']:5.1f}s")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
