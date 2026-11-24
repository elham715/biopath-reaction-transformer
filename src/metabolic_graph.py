"""Metabolic graph topology and Shortest-Path Distance (SPD) computation."""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from .config import (
    METABOLITE_REGISTRY,
    METABOLITE_NAMES,
    ENZYME_REGISTRY,
    ENZYME_NAMES,
    CANONICAL_REACTIONS,
    NUM_METABOLITES,
    NUM_ENZYMES
)

class MetabolicGraph:
    """Directed graph representation of human biochemical reactions and topological distances."""
    
    def __init__(self, max_spd: int = 8):
        self.max_spd = max_spd
        self.metabolite_to_id = {name: idx for idx, name in enumerate(METABOLITE_NAMES)}
        self.id_to_metabolite = {idx: name for idx, name in enumerate(METABOLITE_NAMES)}
        self.num_nodes = len(METABOLITE_NAMES)
        
        # Build adjacency matrix
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int32)
        self.edge_types = {}  # (u, v) -> enzyme_name
        
        for rxn in CANONICAL_REACTIONS:
            sub = rxn["substrate"]
            prod = rxn["product"]
            if sub in self.metabolite_to_id and prod in self.metabolite_to_id:
                u = self.metabolite_to_id[sub]
                v = self.metabolite_to_id[prod]
                self.adj_matrix[u, v] = 1
                self.edge_types[(u, v)] = rxn["enzyme"]
                
        # Compute All-Pairs Shortest Path Distance via Floyd-Warshall
        self.spd_matrix = self._compute_all_pairs_spd()

    def _compute_all_pairs_spd(self) -> np.ndarray:
        """Computes all-pairs shortest path distance matrix bounded by max_spd."""
        inf_val = 9999
        dist = np.full((self.num_nodes, self.num_nodes), inf_val, dtype=np.int32)
        
        # Diagonal is 0
        np.fill_diagonal(dist, 0)
        
        # Direct edges have distance 1 (treating metabolic network as undirected for topological proximity)
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if self.adj_matrix[u, v] == 1 or self.adj_matrix[v, u] == 1:
                    dist[u, v] = 1
                    dist[v, u] = 1
                    
        # Floyd-Warshall DP
        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        
        # Truncate at max_spd (unreachable or > max_spd gets max_spd)
        dist[dist > self.max_spd] = self.max_spd
        return dist

    def get_spd_matrix(self) -> np.ndarray:
        """Returns the [num_nodes, num_nodes] topological distance matrix."""
        return self.spd_matrix

    def get_neighbors(self, metabolite: str) -> List[str]:
        """Returns neighboring metabolites within 1 reaction hop."""
        if metabolite not in self.metabolite_to_id:
            return []
        u = self.metabolite_to_id[metabolite]
        neighbors = []
        for v in range(self.num_nodes):
            if self.adj_matrix[u, v] == 1 or self.adj_matrix[v, u] == 1:
                neighbors.append(self.id_to_metabolite[v])
        return neighbors
