"""Metabolic entity tokenizer for metabolites and enzyme reactions."""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from .config import METABOLITE_NAMES, ENZYME_NAMES
from .metabolic_graph import MetabolicGraph

class MetabolicTokenizer:
    """Tokenizes biochemical entities and produces subgraph topological distance matrices."""
    
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_nodes: int = 32, max_spd: int = 8):
        self.max_nodes = max_nodes
        self.max_spd = max_spd
        
        # Build vocabulary
        self.vocab = [self.PAD_TOKEN, self.CLS_TOKEN, self.UNK_TOKEN] + METABOLITE_NAMES
        self.metabolite_to_id = {name: idx for idx, name in enumerate(self.vocab)}
        self.id_to_metabolite = {idx: name for idx, name in enumerate(self.vocab)}
        
        self.enzyme_to_id = {name: idx + 1 for idx, name in enumerate(ENZYME_NAMES)}
        self.enzyme_to_id[self.PAD_TOKEN] = 0
        
        self.graph = MetabolicGraph(max_spd=max_spd)
        self.global_spd = self.graph.get_spd_matrix()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_enzymes(self) -> int:
        return len(self.enzyme_to_id)

    def encode_pathway(
        self,
        active_metabolites: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes an ordered list of active metabolites along with their pairwise SPD matrix.
        
        Returns:
            token_ids: Tensor of shape [max_nodes]
            mask: BoolTensor of shape [max_nodes]
            spd_submatrix: LongTensor of shape [max_nodes, max_nodes]
        """
        tokens = [self.metabolite_to_id[self.CLS_TOKEN]]
        valid_indices = []
        
        for m in active_metabolites:
            tid = self.metabolite_to_id.get(m, self.metabolite_to_id[self.UNK_TOKEN])
            tokens.append(tid)
            # Find index in global graph
            if m in self.graph.metabolite_to_id:
                valid_indices.append(self.graph.metabolite_to_id[m])
            else:
                valid_indices.append(-1)
            if len(tokens) >= self.max_nodes:
                break
                
        seq_len = len(tokens)
        mask = [True] * seq_len
        
        # Padding
        if seq_len < self.max_nodes:
            pad_len = self.max_nodes - seq_len
            tokens.extend([self.metabolite_to_id[self.PAD_TOKEN]] * pad_len)
            mask.extend([False] * pad_len)
            
        # Build SPD distance matrix: [max_nodes, max_nodes]
        spd = np.full((self.max_nodes, self.max_nodes), self.max_spd, dtype=np.int64)
        np.fill_diagonal(spd, 0)
        
        # Fill known pairwise distances between active metabolites
        # Offset by 1 for CLS token at position 0
        for i in range(len(valid_indices)):
            for j in range(len(valid_indices)):
                g_i = valid_indices[i]
                g_j = valid_indices[j]
                if g_i >= 0 and g_j >= 0:
                    spd[i + 1, j + 1] = self.global_spd[g_i, g_j]
                    
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(spd, dtype=torch.long)
        )
