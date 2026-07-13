"""
network.py - Phenotype (neural network) construction from a genome.

We build a feed-forward network by topologically sorting nodes by their `layer`
field, and we evaluate activations in that order. For recurrent NEAT this would
need to change, but for RL tasks like CartPole/MountainCar feed-forward is
sufficient and faster.

For performance, we cache the sorted node order and a list of (in_node, weight,
out_node) tuples so each forward pass is a tight loop.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .genome import Genome, NodeType, apply_activation


class Network:
    """Feed-forward phenotype built from a Genome. Forward-passes use numpy."""

    def __init__(self, genome: Genome, input_ids: List[int], output_ids: List[int]):
        self.genome = genome
        self.input_ids = input_ids
        self.output_ids = output_ids
        # Forward evaluation order: topologically sorted hidden + output nodes
        self._eval_order: List[int] = []
        self._incoming: dict = {}  # out_id -> list of (in_id, weight, enabled)
        self._build()

    def _build(self) -> None:
        g = self.genome
        # incoming adjacency
        incoming = {nid: [] for nid in g.node_genes}
        for c in g.conn_genes.values():
            incoming[c.out_node].append((c.in_node, c.weight, c.enabled))
        self._incoming = incoming
        # Evaluation order: sort by layer then by node_id (stable)
        nodes = sorted(g.node_genes.values(), key=lambda n: (n.layer, n.node_id))
        self._eval_order = [n.node_id for n in nodes
                            if n.node_type in (NodeType.HIDDEN, NodeType.OUTPUT)]

    def forward(self, obs: np.ndarray) -> np.ndarray:
        g = self.genome
        activations = {}
        for i, nid in enumerate(self.input_ids):
            activations[nid] = float(obs[i])

        for nid in self._eval_order:
            n = g.node_genes[nid]
            total = n.bias
            for in_id, w, en in self._incoming[nid]:
                if not en:
                    continue
                a = activations.get(in_id)
                if a is None:
                    # Input that didn't arrive (shouldn't happen in feed-forward)
                    continue
                total += w * a
            activations[nid] = n.response * apply_activation(total, n.activation)

        return np.array([activations[oid] for oid in self.output_ids], dtype=np.float64)

    # --- vectorised evaluation over a batch (for fast rollouts) -------------
    def forward_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        out = np.empty((len(obs_batch), len(self.output_ids)), dtype=np.float64)
        for i in range(len(obs_batch)):
            out[i] = self.forward(obs_batch[i])
        return out
