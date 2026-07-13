"""
compatibility.py - Speciation & compatibility distance.

The compatibility distance delta = c1 * E / N + c2 * D / N + c3 * W_bar
where E = excess genes, D = disjoint genes, W_bar = mean weight difference of
matching genes, N = genome length (max of the two). We cache by genome size to
avoid pathological blowups.

Two genomes are in the same species if delta < threshold.
"""
from __future__ import annotations

from typing import List, Tuple

from .genome import Genome


def compatibility_distance(g1: Genome, g2: Genome,
                           c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
    """Compute NEAT compatibility distance. Innovations are sorted integers."""
    innovs1 = sorted(g1.conn_genes.keys())
    innovs2 = sorted(g2.conn_genes.keys())
    if not innovs1 and not innovs2:
        return 0.0
    max1 = innovs1[-1] if innovs1 else 0
    max2 = innovs2[-1] if innovs2 else 0
    n1 = len(innovs1)
    n2 = len(innovs2)
    N = max(n1, n2)
    if N < 1:
        N = 1
    # Excess: trailing innovs of the longer one not present in shorter
    excess = 0
    disjoint = 0
    matching_w_diff = []
    i = j = 0
    while i < n1 and j < n2:
        a = innovs1[i]
        b = innovs2[j]
        if a == b:
            matching_w_diff.append(abs(g1.conn_genes[a].weight - g2.conn_genes[b].weight))
            i += 1
            j += 1
        elif a < b:
            disjoint += 1
            i += 1
        else:
            disjoint += 1
            j += 1
    # Remaining are excess
    excess += (n1 - i) + (n2 - j)
    w_bar = sum(matching_w_diff) / max(1, len(matching_w_diff))
    # Use a small-N normalization: many NEAT impls divide by 1 if N < 20.
    if N < 20:
        N = 1
    delta = (c1 * excess) / N + (c2 * disjoint) / N + c3 * w_bar
    return delta


class Species:
    def __init__(self, sid: int, representative: Genome):
        self.id = sid
        self.representative = representative
        self.members: List[Genome] = []
        self.best_fitness: float = -1e9
        self.best_genome: Genome | None = None
        self.staleness: int = 0  # generations since improvement
        self.avg_fitness: float = 0.0

    def add(self, g: Genome) -> None:
        g.species_id = self.id
        self.members.append(g)

    def reset(self) -> None:
        self.members = []


class Speciator:
    """Manages species across generations."""

    def __init__(self, threshold: float = 3.0,
                 c1: float = 1.0, c2: float = 1.0, c3: float = 0.4,
                 threshold_adjust: bool = True,
                 target_species: int = 10,
                 threshold_floor: float = 1.0,
                 threshold_ceiling: float = 10.0):
        self.threshold = threshold
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.threshold_adjust = threshold_adjust
        self.target_species = target_species
        self.threshold_floor = threshold_floor
        self.threshold_ceiling = threshold_ceiling
        self._next_sid = 0
        self.species: List[Species] = []

    def _new_sid(self) -> int:
        s = self._next_sid
        self._next_sid += 1
        return s

    def speciate(self, population: List[Genome], rng) -> None:
        # Clear members of existing species
        for sp in self.species:
            sp.reset()
        # If no species yet, seed with first genome
        if not self.species:
            sp = Species(self._new_sid(), population[0])
            sp.add(population[0])
            self.species.append(sp)
            rest = population[1:]
        else:
            rest = list(population)

        for g in rest:
            placed = False
            for sp in self.species:
                rep = sp.representative
                d = compatibility_distance(g, rep, self.c1, self.c2, self.c3)
                if d < self.threshold:
                    sp.add(g)
                    placed = True
                    break
            if not placed:
                sp = Species(self._new_sid(), g)
                sp.add(g)
                self.species.append(sp)

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Update representatives: pick a random member as next rep
        for sp in self.species:
            sp.representative = rng.choice(sp.members)

        # Dynamic threshold adjustment
        if self.threshold_adjust:
            if len(self.species) > self.target_species:
                self.threshold = min(self.threshold_ceiling, self.threshold + 0.3)
            elif len(self.species) < self.target_species // 2 + 1:
                self.threshold = max(self.threshold_floor, self.threshold - 0.3)

    def compute_adjusted_fitness(self) -> None:
        """Explicit fitness sharing: f_i' = f_i / |species|."""
        for sp in self.species:
            n = len(sp.members)
            if n == 0:
                continue
            total = 0.0
            for g in sp.members:
                g.adjusted_fitness = g.fitness / n
                total += g.adjusted_fitness
            sp.avg_fitness = total / n
