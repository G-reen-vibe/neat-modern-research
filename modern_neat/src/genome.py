"""
genome.py - Core genome representation for neural networks evolved via NEAT-like algorithms.

A genome encodes:
- nodes (neurons) with their type (input/hidden/output), activation function, and layer
- connections (synapses) with innovation number, in/out node, weight, enabled flag

This representation is shared across all algorithm variants in this project so that
we can cleanly swap evolutionary strategies without changing the substrate.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class Activation(Enum):
    """Activation functions. Named rather than function pointers so genomes stay
    serialisable and we can audit which functions are most useful across runs."""
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    GELU = "gelu"
    SIN = "sin"            # periodic - good for some control tasks
    STEP = "step"          # classic perceptron
    LRELU = "lrelu"        # leaky relu


def apply_activation(x: float, act: Activation) -> float:
    if act is Activation.TANH:
        return math.tanh(x)
    if act is Activation.RELU:
        return x if x > 0.0 else 0.0
    if act is Activation.SIGMOID:
        return 1.0 / (1.0 + math.exp(-x) if x > -60 else 1e26)
    if act is Activation.LINEAR:
        return x
    if act is Activation.GELU:
        # Approximate GELU
        return 0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    if act is Activation.SIN:
        return math.sin(x)
    if act is Activation.STEP:
        return 1.0 if x > 0.0 else 0.0
    if act is Activation.LRELU:
        return x if x > 0.0 else 0.01 * x
    raise ValueError(f"unknown activation {act}")


@dataclass
class NodeGene:
    node_id: int
    node_type: NodeType
    activation: Activation = Activation.TANH
    bias: float = 0.0
    # layer is a continuous scalar used to order feed-forward evaluation.
    # Input nodes have layer=0, output nodes layer=1; hidden nodes get
    # fractional layers so we can have multiple "rings" of hidden units.
    layer: float = 0.5
    response: float = 1.0  # multiplicative gain - lets evolution scale activation


@dataclass
class ConnectionGene:
    innov: int           # innovation number, used for compatibility distance
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    node_genes: Dict[int, NodeGene] = field(default_factory=dict)
    conn_genes: Dict[int, ConnectionGene] = field(default_factory=dict)  # keyed by innov
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    # Multi-objective: track behavioural descriptor and novelty separately
    behavior: Optional[Tuple[float, ...]] = None
    novelty: float = 0.0
    age: int = 0
    species_id: int = -1

    # --- structural helpers -------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.node_genes)

    @property
    def num_enabled_conns(self) -> int:
        return sum(1 for c in self.conn_genes.values() if c.enabled)

    @property
    def num_conns(self) -> int:
        return len(self.conn_genes)

    def add_node(self, node: NodeGene) -> None:
        self.node_genes[node.node_id] = node

    def add_connection(self, conn: ConnectionGene) -> None:
        self.conn_genes[conn.innov] = conn

    def has_connection(self, in_node: int, out_node: int) -> Optional[ConnectionGene]:
        for c in self.conn_genes.values():
            if c.in_node == in_node and c.out_node == out_node:
                return c
        return None

    def is_feedforward(self) -> bool:
        """A genome is feed-forward iff there are no cycles in the connection graph
        respecting layer ordering. We allow only conns where layer(out) > layer(in)."""
        for c in self.conn_genes.values():
            if not c.enabled:
                continue
            in_layer = self.node_genes[c.in_node].layer
            out_layer = self.node_genes[c.out_node].layer
            if out_layer <= in_layer:
                return False
        return True

    # --- copy ----------------------------------------------------------------
    def copy(self) -> "Genome":
        g = Genome()
        for nid, n in self.node_genes.items():
            g.node_genes[nid] = NodeGene(
                node_id=n.node_id,
                node_type=n.node_type,
                activation=n.activation,
                bias=n.bias,
                layer=n.layer,
                response=n.response,
            )
        for inv, c in self.conn_genes.items():
            g.conn_genes[inv] = ConnectionGene(
                innov=c.innov,
                in_node=c.in_node,
                out_node=c.out_node,
                weight=c.weight,
                enabled=c.enabled,
            )
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        g.behavior = self.behavior
        g.novelty = self.novelty
        g.age = self.age
        g.species_id = self.species_id
        return g


def make_initial_genome(
    n_inputs: int,
    n_outputs: int,
    rng: random.Random,
    weight_init: str = "uniform",
    weight_scale: float = 1.0,
    output_activation: Activation = Activation.TANH,
) -> Genome:
    """Minimal genome: only inputs -> outputs, fully connected."""
    g = Genome()
    for i in range(n_inputs):
        g.add_node(NodeGene(node_id=i + 1, node_type=NodeType.INPUT,
                            activation=Activation.LINEAR, layer=0.0))
    for o in range(n_outputs):
        g.add_node(NodeGene(node_id=n_inputs + o + 1, node_type=NodeType.OUTPUT,
                            activation=output_activation, layer=1.0))

    # Fully connect inputs to outputs
    innov = 1
    for i in range(1, n_inputs + 1):
        for o in range(n_inputs + 1, n_inputs + n_outputs + 1):
            w = rng.uniform(-weight_scale, weight_scale) if weight_init == "uniform" \
                else rng.gauss(0, weight_scale)
            g.add_connection(ConnectionGene(innov=innov, in_node=i, out_node=o,
                                            weight=w))
            innov += 1
    return g
