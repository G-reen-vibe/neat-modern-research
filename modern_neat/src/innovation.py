"""
innovation.py - Global innovation bookkeeping.

In classic NEAT, every structural mutation (a new node split, or a new connection)
gets a global "innovation number" so that two genomes that make the same structural
change end up with matching genes - this is what makes the compatibility distance
metric meaningful.

We keep a process-wide InnovationStore so that all genomes in a generation share
the same numbering. This matches the canonical NEAT implementation.
"""
from __future__ import annotations

import random
from threading import Lock
from typing import Dict, Tuple


class InnovationStore:
    """Assigns consistent innovation numbers to structural mutations."""

    def __init__(self, start_innov: int = 1):
        self._next_innov = start_innov
        self._next_node_id = 1
        self._conn_history: Dict[Tuple[int, int], int] = {}   # (in,out) -> innov
        self._node_history: Dict[int, Tuple[int, int, int]] = {}  # innov -> (in, out, new_id)
        self._lock = Lock()

    def next_innov(self) -> int:
        with self._lock:
            i = self._next_innov
            self._next_innov += 1
            return i

    def next_node_id(self) -> int:
        with self._lock:
            i = self._next_node_id
            self._next_node_id += 1
            return i

    def get_connection_innov(self, in_node: int, out_node: int) -> int:
        """Return the innovation number for (in,out), creating one if needed."""
        key = (in_node, out_node)
        with self._lock:
            if key in self._conn_history:
                return self._conn_history[key]
            i = self._next_innov
            self._next_innov += 1
            self._conn_history[key] = i
            return i

    def get_node_split_innov(self, conn_innov: int, in_node: int, out_node: int,
                              rng: random.Random) -> Tuple[int, int, int]:
        """For an add-node mutation that splits connection conn_innov, return
        (new_node_id, innov_in_to_new, innov_new_to_out). If the same split
        has happened before (tracked by conn_innov), reuse IDs."""
        with self._lock:
            if conn_innov in self._node_history:
                return self._node_history[conn_innov]
            new_id = self._next_node_id
            self._next_node_id += 1
            i1 = self._next_innov
            self._next_innov += 1
            i2 = self._next_innov
            self._next_innov += 1
            self._node_history[conn_innov] = (new_id, i1, i2)
            return new_id, i1, i2
