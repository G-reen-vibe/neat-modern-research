"""
mutations.py - Structural and weight mutation operators for NEAT genomes.

These are the standard NEAT mutation operators, implemented cleanly. Variants of
the algorithm (different selection strategies, novelty, CMA-ES weight training,
etc.) can mix-and-match these.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from .genome import (Activation, ConnectionGene, Genome, NodeGene, NodeType,
                     apply_activation)
from .innovation import InnovationStore


def mutate_weights(genome: Genome, rng: random.Random,
                   rate: float = 0.8,
                   perturb_prob: float = 0.9,
                   perturb_step: float = 0.2,
                   reset_scale: float = 1.0) -> None:
    """Gaussian-perturb each weight with prob `rate`. With prob 1-perturb_prob,
    assign a new random value."""
    for c in genome.conn_genes.values():
        if rng.random() < rate:
            if rng.random() < perturb_prob:
                c.weight += rng.gauss(0, perturb_step)
            else:
                c.weight = rng.gauss(0, reset_scale)
            # clip to keep numerics sane
            if c.weight > 5.0:
                c.weight = 5.0
            elif c.weight < -5.0:
                c.weight = -5.0


def mutate_biases(genome: Genome, rng: random.Random,
                  rate: float = 0.8,
                  perturb_prob: float = 0.9,
                  perturb_step: float = 0.2,
                  reset_scale: float = 1.0) -> None:
    """Same as mutate_weights but for node biases."""
    for n in genome.node_genes.values():
        if n.node_type is NodeType.INPUT:
            continue
        if rng.random() < rate:
            if rng.random() < perturb_prob:
                n.bias += rng.gauss(0, perturb_step)
            else:
                n.bias = rng.gauss(0, reset_scale)
            if n.bias > 5.0:
                n.bias = 5.0
            elif n.bias < -5.0:
                n.bias = -5.0


def mutate_add_node(genome: Genome, innov_store: InnovationStore,
                    rng: random.Random) -> bool:
    """Split a random enabled connection in two, inserting a hidden node.
    Standard NEAT: new connection in->new gets weight=1, new->out gets old weight;
    old connection is disabled."""
    enabled_conns = [c for c in genome.conn_genes.values() if c.enabled]
    if not enabled_conns:
        return False
    c = rng.choice(enabled_conns)
    new_id, i1, i2 = innov_store.get_node_split_innov(c.innov, c.in_node, c.out_node, rng)
    # If this exact split was already made elsewhere in the population, we still
    # need to check whether the genome already has these conns (rare).
    c.enabled = False
    in_layer = genome.node_genes[c.in_node].layer
    out_layer = genome.node_genes[c.out_node].layer
    mid_layer = (in_layer + out_layer) / 2.0
    # Activation: random selection biased toward tanh / relu which we found most useful
    act = rng.choice([Activation.TANH, Activation.RELU, Activation.GELU, Activation.LRELU])
    genome.add_node(NodeGene(node_id=new_id, node_type=NodeType.HIDDEN,
                             activation=act, layer=mid_layer))
    genome.add_connection(ConnectionGene(innov=i1, in_node=c.in_node, out_node=new_id,
                                          weight=1.0))
    genome.add_connection(ConnectionGene(innov=i2, in_node=new_id, out_node=c.out_node,
                                          weight=c.weight))
    return True


def mutate_add_connection(genome: Genome, innov_store: InnovationStore,
                          rng: random.Random,
                          max_tries: int = 20,
                          weight_scale: float = 1.0) -> bool:
    """Add a new feed-forward connection between two previously unconnected nodes."""
    nodes = list(genome.node_genes.values())
    if len(nodes) < 2:
        return False
    for _ in range(max_tries):
        a = rng.choice(nodes)
        b = rng.choice(nodes)
        # enforce feedforward: layer(b) > layer(a); b not input
        if b.node_type is NodeType.INPUT:
            continue
        if a.node_type is NodeType.OUTPUT:
            continue
        if b.layer <= a.layer:
            continue
        if a.node_id == b.node_id:
            continue
        # already exists?
        if genome.has_connection(a.node_id, b.node_id):
            continue
        innov = innov_store.get_connection_innov(a.node_id, b.node_id)
        genome.add_connection(ConnectionGene(innov=innov, in_node=a.node_id,
                                              out_node=b.node_id,
                                              weight=rng.gauss(0, weight_scale)))
        return True
    return False


def mutate_toggle_enable(genome: Genome, rng: random.Random,
                         rate: float = 0.01,
                         enable_only: bool = False) -> None:
    """Randomly toggle the enabled flag of connections."""
    for c in genome.conn_genes.values():
        if enable_only:
            if not c.enabled and rng.random() < rate:
                c.enabled = True
        else:
            if rng.random() < rate:
                c.enabled = not c.enabled


def mutate_activation(genome: Genome, rng: random.Random, rate: float = 0.1) -> None:
    """Randomly change a hidden node's activation function."""
    hidden = [n for n in genome.node_genes.values() if n.node_type is NodeType.HIDDEN]
    if not hidden:
        return
    n = rng.choice(hidden)
    if rng.random() < rate:
        n.activation = rng.choice([Activation.TANH, Activation.RELU, Activation.GELU,
                                   Activation.LRELU, Activation.SIGMOID, Activation.SIN])


def mutate_response(genome: Genome, rng: random.Random,
                    rate: float = 0.1, perturb: float = 0.1) -> None:
    """Perturb the multiplicative gain (response) of hidden nodes."""
    for n in genome.node_genes.values():
        if n.node_type is NodeType.INPUT:
            continue
        if rng.random() < rate:
            n.response = max(0.05, n.response + rng.gauss(0, perturb))
