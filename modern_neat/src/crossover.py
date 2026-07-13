"""
crossover.py - Standard NEAT crossover operator.

Crossover(a, b) where a is the more-fit parent:
- For each innovation in either parent:
  - if matching: inherit from either parent (random), but skip disabled if either
    parent disabled it with some probability
  - if disjoint/excess: inherit from more-fit parent
- Nodes: union of nodes referenced by the inherited connections (plus inputs/outputs)
"""
from __future__ import annotations

import random

from .genome import (Activation, ConnectionGene, Genome, NodeGene, NodeType)
from .innovation import InnovationStore


def crossover(parent_a: Genome, parent_b: Genome, rng: random.Random,
              innov_store: InnovationStore,
              mate_avg_prob: float = 0.5,
              disable_if_either_disabled_prob: float = 0.75) -> Genome:
    """Parent A is assumed to be more-fit (caller's responsibility)."""
    if parent_b.fitness > parent_a.fitness:
        parent_a, parent_b = parent_b, parent_a

    child = Genome()
    # Inherit connections
    a_innovs = set(parent_a.conn_genes.keys())
    b_innovs = set(parent_b.conn_genes.keys())
    all_innovs = sorted(a_innovs | b_innovs)
    for inv in all_innovs:
        if inv in a_innovs and inv in b_innovs:
            # matching: pick either
            src = parent_a if rng.random() < mate_avg_prob else parent_b
            c = parent_a.conn_genes[inv]
            c_b = parent_b.conn_genes[inv]
            enabled = c.enabled and c_b.enabled
            if not enabled:
                # if either is disabled, child gets disabled with prob
                if rng.random() < disable_if_either_disabled_prob:
                    enabled = False
                else:
                    enabled = True
            # Average the weights half the time, otherwise inherit from src
            if rng.random() < 0.5:
                w = 0.5 * (c.weight + c_b.weight)
            else:
                w = src.conn_genes[inv].weight
            child.add_connection(ConnectionGene(innov=inv,
                                                 in_node=c.in_node,
                                                 out_node=c.out_node,
                                                 weight=w,
                                                 enabled=enabled))
        elif inv in a_innovs:
            c = parent_a.conn_genes[inv]
            child.add_connection(ConnectionGene(innov=inv,
                                                 in_node=c.in_node,
                                                 out_node=c.out_node,
                                                 weight=c.weight,
                                                 enabled=c.enabled))
        # parent_b-only genes are dropped (since a is more fit)

    # Inherit nodes: union referenced, prefer A's metadata
    needed = set()
    for c in child.conn_genes.values():
        needed.add(c.in_node)
        needed.add(c.out_node)
    # Always keep inputs and outputs
    for nid, n in parent_a.node_genes.items():
        if n.node_type in (NodeType.INPUT, NodeType.OUTPUT):
            child.add_node(NodeGene(node_id=n.node_id, node_type=n.node_type,
                                     activation=n.activation, bias=n.bias,
                                     layer=n.layer, response=n.response))
    # Hidden nodes from A
    for nid, n in parent_a.node_genes.items():
        if n.node_type is NodeType.HIDDEN and nid in needed:
            # If B also has it, average bias
            if nid in parent_b.node_genes:
                nb = parent_b.node_genes[nid]
                bias = 0.5 * (n.bias + nb.bias) if rng.random() < 0.5 else n.bias
                response = 0.5 * (n.response + nb.response) if rng.random() < 0.5 else n.response
                act = n.activation if rng.random() < 0.5 else nb.activation
            else:
                bias, response, act = n.bias, n.response, n.activation
            child.add_node(NodeGene(node_id=n.node_id, node_type=NodeType.HIDDEN,
                                     activation=act, bias=bias,
                                     layer=n.layer, response=response))
    return child
