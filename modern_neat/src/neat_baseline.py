"""
neat_baseline.py - Canonical NEAT (Stanley & Miikkulainen 2002).

This is the reference implementation we'll be improving on. Configuration is
deliberately close to the original paper so that the baseline is faithful.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .compatibility import Species, Speciator, compatibility_distance
from .crossover import crossover
from .genome import (Activation, ConnectionGene, Genome, NodeGene, NodeType,
                     make_initial_genome)
from .innovation import InnovationStore
from .mutations import (mutate_add_connection, mutate_add_node, mutate_activation,
                        mutate_biases, mutate_response, mutate_toggle_enable,
                        mutate_weights)
from .network import Network


@dataclass
class NEATConfig:
    # Population
    pop_size: int = 150
    # Mutation rates
    weight_mut_rate: float = 0.8
    weight_perturb_prob: float = 0.9
    weight_perturb_step: float = 0.2
    bias_mut_rate: float = 0.8
    bias_perturb_prob: float = 0.9
    bias_perturb_step: float = 0.2
    add_node_rate: float = 0.03
    add_conn_rate: float = 0.05
    toggle_enable_rate: float = 0.01
    activation_mut_rate: float = 0.0
    response_mut_rate: float = 0.0
    # Crossover / mating
    mate_avg_prob: float = 0.5
    disable_if_either_disabled_prob: float = 0.75
    interspecies_mate_rate: float = 0.001
    # Selection
    elitism_per_species: int = 1
    survival_threshold: float = 0.2   # fraction of each species that survives
    min_species_size: int = 2
    # Speciation
    compat_threshold: float = 3.0
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    target_species: int = 10
    threshold_adjust: bool = True
    # Reproduction
    reproduce_asexual_rate: float = 0.5
    # Reset
    weight_reset_scale: float = 1.0
    bias_reset_scale: float = 1.0
    # Initialization
    weight_init_scale: float = 1.0
    output_activation: Activation = Activation.TANH
    # Novelty
    use_novelty: bool = False
    novelty_weight: float = 0.0
    archive_size: int = 200
    # Bookkeeping
    n_episodes: int = 1
    max_steps: int = 1000
    seed_offset: int = 0
    # Stagnation
    max_staleness: int = 15
    # Behavior diversity (for novelty)
    n_behavior_episodes: int = 1
    behavior_max_steps: int = 200


class NEATBaseline:
    """Vanilla NEAT. Designed to be subclassed by modern variants."""

    def __init__(self, env_id: str, n_inputs: int, n_outputs: int,
                 config: NEATConfig, seed: int = 0, log_dir: Optional[str] = None):
        self.env_id = env_id
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.innov_store = InnovationStore(start_innov=1)
        # IMPORTANT: node ids need to start after the initial input/output nodes
        self.innov_store._next_node_id = n_inputs + n_outputs + 1
        self.speciator = Speciator(
            threshold=config.compat_threshold,
            c1=config.c1, c2=config.c2, c3=config.c3,
            threshold_adjust=config.threshold_adjust,
            target_species=config.target_species,
        )
        self.population: List[Genome] = []
        self.generation: int = 0
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -1e9
        self.history: List[dict] = []
        self.log_dir = log_dir
        # Novelty archive
        self.archive: List[np.ndarray] = []
        self._init_population()

    # ------------------------------------------------------------------ init
    def _init_population(self) -> None:
        for _ in range(self.config.pop_size):
            g = make_initial_genome(self.n_inputs, self.n_outputs, self.rng,
                                     weight_scale=self.config.weight_init_scale,
                                     output_activation=self.config.output_activation)
            self.population.append(g)

    # ----------------------------------------------------------- evaluation
    def build_network(self, g: Genome) -> Network:
        input_ids = list(range(1, self.n_inputs + 1))
        output_ids = list(range(self.n_inputs + 1, self.n_inputs + self.n_outputs + 1))
        return Network(g, input_ids, output_ids)

    def evaluate_population(self) -> None:
        for g in self.population:
            net = self.build_network(g)
            from .env_wrapper import evaluate_genome
            r, _ = evaluate_genome(net, self.env_id,
                                    n_episodes=self.config.n_episodes,
                                    max_steps=self.config.max_steps,
                                    seed_offset=self.config.seed_offset + self.generation * 100)
            g.fitness = r
            if self.config.use_novelty and self.config.novelty_weight > 0:
                from .env_wrapper import collect_behavior_descriptor
                _, desc, _ = collect_behavior_descriptor(
                    net, self.env_id,
                    n_episodes=self.config.n_behavior_episodes,
                    max_steps=self.config.behavior_max_steps,
                    seed_offset=self.config.seed_offset + self.generation * 100)
                g.behavior = tuple(desc)
                g.novelty = self._compute_novelty(desc)

    def _compute_novelty(self, desc: np.ndarray) -> float:
        if not self.archive and not self.population:
            return 0.0
        all_descs = list(self.archive) + [g.behavior for g in self.population
                                           if g.behavior is not None]
        all_descs = [d for d in all_descs if d is not None]
        if not all_descs:
            return 0.0
        arr = np.array(all_descs)
        d = np.linalg.norm(arr - desc, axis=1)
        d.sort()
        k = min(15, len(d))
        return float(d[:k].mean())

    def _maybe_archive(self, desc: np.ndarray, threshold: float = 0.05) -> None:
        if not self.archive:
            self.archive.append(desc)
            return
        arr = np.array(self.archive)
        d = np.linalg.norm(arr - desc, axis=1)
        if d.min() > threshold:
            self.archive.append(desc)
            if len(self.archive) > self.config.archive_size:
                self.archive = self.archive[-self.config.archive_size:]

    # ----------------------------------------------------------- speciation
    def speciate(self) -> None:
        self.speciator.speciate(self.population, self.rng)
        self.speciator.compute_adjusted_fitness()

    # ----------------------------------------------------------- selection
    def _select_parent(self, species: Species) -> Genome:
        # Tournament of 3 within species
        candidates = self.rng.sample(species.members, k=min(3, len(species.members)))
        return max(candidates, key=lambda g: g.adjusted_fitness)

    def _cull_species(self) -> None:
        for sp in self.speciator.species:
            sp.members.sort(key=lambda g: g.fitness, reverse=True)
            # Update best
            if sp.members[0].fitness > sp.best_fitness:
                sp.best_fitness = sp.members[0].fitness
                sp.best_genome = sp.members[0].copy()
                sp.staleness = 0
            else:
                sp.staleness += 1
            # Cull bottom (1 - survival_threshold)
            surviving = max(self.config.min_species_size,
                              int(len(sp.members) * self.config.survival_threshold))
            sp.members = sp.members[:surviving]

    # ----------------------------------------------------------- reproduction
    def _allocate_offspring(self) -> Dict[int, int]:
        """Allocate offspring to species proportional to adjusted fitness sum."""
        species_fitness = {}
        total = 0.0
        for sp in self.speciator.species:
            sf = sum(g.adjusted_fitness for g in sp.members)
            species_fitness[sp.id] = max(0.0, sf)
            total += sf
        # In case of all-zero fitness, split equally
        if total == 0:
            n_sp = len(self.speciator.species)
            return {sp.id: max(1, self.config.pop_size // n_sp)
                    for sp in self.speciator.species}
        allocation = {}
        for sp in self.speciator.species:
            allocation[sp.id] = int(round(species_fitness[sp.id] / total *
                                            self.config.pop_size))
        # Fix rounding
        diff = self.config.pop_size - sum(allocation.values())
        if diff != 0:
            keys = list(allocation.keys())
            for i in range(abs(diff)):
                allocation[keys[i % len(keys)]] += 1 if diff > 0 else -1
                if allocation[keys[i % len(keys)]] < 0:
                    allocation[keys[i % len(keys)]] = 0
                    diff += 1
        return allocation

    def _reproduce(self) -> List[Genome]:
        new_pop: List[Genome] = []
        # Filter stagnant species (except keep top one if all are stagnant)
        all_stagnant = all(sp.staleness >= self.config.max_staleness
                            for sp in self.speciator.species)
        active_species = []
        for sp in self.speciator.species:
            if all_stagnant or sp.staleness < self.config.max_staleness:
                active_species.append(sp)
        if not active_species:
            active_species = self.speciator.species
        allocation = self._allocate_offspring_filtered(active_species)

        for sp in active_species:
            n = allocation.get(sp.id, 0)
            if n <= 0:
                continue
            # Elitism: copy best genome unchanged
            for e in range(min(self.config.elitism_per_species, len(sp.members))):
                new_pop.append(sp.members[e].copy())
            # Generate rest
            n_remaining = n - min(self.config.elitism_per_species, len(sp.members))
            for _ in range(n_remaining):
                if self.rng.random() < self.config.reproduce_asexual_rate:
                    parent = self._select_parent(sp)
                    child = parent.copy()
                else:
                    p1 = self._select_parent(sp)
                    if self.rng.random() < self.config.interspecies_mate_rate \
                            and len(active_species) > 1:
                        other = self.rng.choice(active_species)
                        p2 = self._select_parent(other)
                    else:
                        p2 = self._select_parent(sp)
                    child = crossover(p1, p2, self.rng, self.innov_store,
                                       mate_avg_prob=self.config.mate_avg_prob,
                                       disable_if_either_disabled_prob=
                                       self.config.disable_if_either_disabled_prob)
                self._mutate(child)
                new_pop.append(child)
        # If we have too many, trim. If too few, top up from random parents.
        while len(new_pop) > self.config.pop_size:
            new_pop.pop()
        while len(new_pop) < self.config.pop_size:
            sp = self.rng.choice(active_species)
            parent = self._select_parent(sp)
            child = parent.copy()
            self._mutate(child)
            new_pop.append(child)
        return new_pop

    def _allocate_offspring_filtered(self, active_species: List[Species]) -> Dict[int, int]:
        species_fitness = {}
        total = 0.0
        for sp in active_species:
            sf = sum(g.adjusted_fitness for g in sp.members)
            species_fitness[sp.id] = max(0.0, sf)
            total += sf
        if total == 0:
            n_sp = len(active_species)
            return {sp.id: max(1, self.config.pop_size // n_sp)
                    for sp in active_species}
        allocation = {}
        for sp in active_species:
            allocation[sp.id] = int(round(species_fitness[sp.id] / total *
                                            self.config.pop_size))
        diff = self.config.pop_size - sum(allocation.values())
        if diff != 0:
            keys = list(allocation.keys())
            for i in range(abs(diff)):
                if diff > 0:
                    allocation[keys[i % len(keys)]] += 1
                else:
                    if allocation[keys[i % len(keys)]] > 0:
                        allocation[keys[i % len(keys)]] -= 1
                    else:
                        diff -= 1
        return allocation

    # ----------------------------------------------------------- mutation
    def _mutate(self, g: Genome) -> None:
        cfg = self.config
        if self.rng.random() < cfg.add_node_rate:
            mutate_add_node(g, self.innov_store, self.rng)
        if self.rng.random() < cfg.add_conn_rate:
            mutate_add_connection(g, self.innov_store, self.rng,
                                   weight_scale=cfg.weight_init_scale)
        mutate_weights(g, self.rng,
                        rate=cfg.weight_mut_rate,
                        perturb_prob=cfg.weight_perturb_prob,
                        perturb_step=cfg.weight_perturb_step,
                        reset_scale=cfg.weight_reset_scale)
        mutate_biases(g, self.rng,
                       rate=cfg.bias_mut_rate,
                       perturb_prob=cfg.bias_perturb_prob,
                       perturb_step=cfg.bias_perturb_step,
                       reset_scale=cfg.bias_reset_scale)
        if cfg.toggle_enable_rate > 0:
            mutate_toggle_enable(g, self.rng, rate=cfg.toggle_enable_rate)
        if cfg.activation_mut_rate > 0:
            mutate_activation(g, self.rng, rate=cfg.activation_mut_rate)
        if cfg.response_mut_rate > 0:
            mutate_response(g, self.rng, rate=cfg.response_mut_rate,
                             perturb=cfg.weight_perturb_step)
        # Age the genome
        g.age = 0

    # ----------------------------------------------------------- main loop
    def step(self) -> dict:
        t0 = time.time()
        self.evaluate_population()
        # Track best
        for g in self.population:
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g.copy()
        self.speciate()
        self._cull_species()
        stats = self._compute_stats()
        new_pop = self._reproduce()
        # Increment age
        for g in new_pop:
            g.age = 0
        # Save old population's age increment for survivors (already reset above)
        self.population = new_pop
        self.generation += 1
        stats["wall_time_s"] = time.time() - t0
        self.history.append(stats)
        return stats

    def _compute_stats(self) -> dict:
        fits = [g.fitness for g in self.population]
        nodes = [g.num_nodes for g in self.population]
        conns = [g.num_conns for g in self.population]
        adj = [g.adjusted_fitness for g in self.population]
        return {
            "generation": self.generation,
            "max_fitness": max(fits),
            "mean_fitness": float(np.mean(fits)),
            "median_fitness": float(np.median(fits)),
            "mean_adj_fitness": float(np.mean(adj)) if adj else 0.0,
            "best_overall": self.best_fitness,
            "mean_nodes": float(np.mean(nodes)),
            "max_nodes": max(nodes),
            "mean_conns": float(np.mean(conns)),
            "max_conns": max(conns),
            "n_species": len(self.speciator.species),
            "species_sizes": [len(s.members) for s in self.speciator.species],
        }
